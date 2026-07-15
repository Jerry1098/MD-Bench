/*
 * Copyright (C)  NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of MD-Bench.
 * Use of this source code is governed by a LGPL-3.0
 * license that can be found in the LICENSE file.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//---

#include <cub/cub.cuh>
#include <device.h>
#include <gpu_profiler.h>

extern "C" {
#include <atom.h>
#include <neighbor.h>
#include <parameter.h>
#include <util.h>
}

extern MD_FLOAT xprd, yprd, zprd;
extern MD_FLOAT bininvx, bininvy, bininvz;
extern int mbinxlo, mbinylo, mbinzlo;
extern int nbinx, nbiny, nbinz;
extern int mbinx, mbiny, mbinz; // n bins in x, y, z
extern int mbins;               // total number of bins
extern int atoms_per_bin;       // max atoms per bin
extern MD_FLOAT cutneighsq;     // neighbor cutoff squared
extern int nmax;
extern int nstencil; // # of bins in stencil
extern int* stencil; // stencil list of bin offsets
static int* c_stencil       = NULL;
static int* c_resize_needed = NULL;
static int* c_new_maxneighs = NULL;
static Binning c_binning {
    .bincount = NULL, .bins = NULL, .mbins = 0, .atoms_per_bin = 0
};
// device-side sort scratch
static int* c_binpos            = NULL;
static void* c_sortScanTemp     = NULL;
static size_t c_sortScanTempCap = 0;
static MD_FLOAT* c_sort_x       = NULL;
static MD_FLOAT* c_sort_vx      = NULL;
static int c_sortNmax           = 0;

// Multi GPU
extern int pad_x, pad_y, pad_z;
extern MD_FLOAT binsizex, binsizey, binsizez;

__device__ int coord2bin_device(
    MD_FLOAT xin, MD_FLOAT yin, MD_FLOAT zin, Neighbor_params np)
{
    /*
    int ix, iy, iz;

    if (xin >= np.xprd) {
        ix = (int)((xin - np.xprd) * np.bininvx) + np.nbinx - np.mbinxlo;
    } else if (xin >= 0.0) {
        ix = (int)(xin * np.bininvx) - np.mbinxlo;
    } else {
        ix = (int)(xin * np.bininvx) - np.mbinxlo - 1;
    }

    if (yin >= np.yprd) {
        iy = (int)((yin - np.yprd) * np.bininvy) + np.nbiny - np.mbinylo;
    } else if (yin >= 0.0) {
        iy = (int)(yin * np.bininvy) - np.mbinylo;
    } else {
        iy = (int)(yin * np.bininvy) - np.mbinylo - 1;
    }

    if (zin >= np.zprd) {
        iz = (int)((zin - np.zprd) * np.bininvz) + np.nbinz - np.mbinzlo;
    } else if (zin >= 0.0) {
        iz = (int)(zin * np.bininvz) - np.mbinzlo;
    } else {
        iz = (int)(zin * np.bininvz) - np.mbinzlo - 1;
    }

    return (iz * np.mbiny * np.mbinx + iy * np.mbinx + ix + 1);
    */

    int ix, iy, iz;
    MD_FLOAT eps = 1e-9;
    MD_FLOAT xlo = 0.0;
    MD_FLOAT ylo = 0.0;
    MD_FLOAT zlo = 0.0;
    xlo          = fabs(xlo - np.pad_x * np.binsizex) + eps;
    ylo          = fabs(ylo - np.pad_y * np.binsizey) + eps;
    zlo          = fabs(zlo - np.pad_z * np.binsizez) + eps;
    ix           = (int)((xin + xlo) * np.bininvx);
    iy           = (int)((yin + ylo) * np.bininvy);
    iz           = (int)((zin + zlo) * np.bininvz);
    return (iz * np.mbiny * np.mbinx + iy * np.mbinx + ix);
}

/* sorts the contents of a bin to make it comparable to the CPU version */
/* uses bubble sort since atoms per bin should be relatively small and can be done in situ
 */
__global__ void sort_bin_contents_kernel(
    int* bincount, int* bins, int mbins, int atoms_per_bin)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mbins) {
        return;
    }

    int atoms_in_bin = bincount[i];
    int* bin_ptr     = &bins[i * atoms_per_bin];
    int sorted;
    do {
        sorted = 1;
        int tmp;
        for (int index = 0; index < atoms_in_bin - 1; index++) {
            if (bin_ptr[index] > bin_ptr[index + 1]) {
                tmp                = bin_ptr[index];
                bin_ptr[index]     = bin_ptr[index + 1];
                bin_ptr[index + 1] = tmp;
                sorted             = 0;
            }
        }
    } while (!sorted);
}

__global__ void binatoms_kernel(DeviceAtom a,
    int nall,
    int* bincount,
    int* bins,
    int atoms_per_bin,
    Neighbor_params np,
    int* resize_needed)
{
    DeviceAtom* atom = &a;
    const int i      = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nall) {
        return;
    }

    MD_FLOAT x = atom_x(i);
    MD_FLOAT y = atom_y(i);
    MD_FLOAT z = atom_z(i);
    int ibin   = coord2bin_device(x, y, z, np);
    int ac     = atomicAdd(&bincount[ibin], 1);

    if (ac < atoms_per_bin) {
        bins[ibin * atoms_per_bin + ac] = i;
    } else {
        atomicMax(resize_needed, ac);
    }
}

__global__ void compute_neighborhood(DeviceAtom a,
    DeviceNeighbor neigh,
    Neighbor_params np,
    int nlocal,
    int maxneighs,
    int halfneigh,
    int nstencil,
    int* stencil,
    int* bins,
    int atoms_per_bin,
    int* bincount,
    int* new_maxneighs,
    MD_FLOAT cutneighsq,
    int ntypes)
{

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) {
        return;
    }

    DeviceAtom* atom         = &a;
    DeviceNeighbor* neighbor = &neigh;

    int n         = 0;
    MD_FLOAT xtmp = atom_x(i);
    MD_FLOAT ytmp = atom_y(i);
    MD_FLOAT ztmp = atom_z(i);
    int ibin      = coord2bin_device(xtmp, ytmp, ztmp, np);
#if LJ_COMB_RULE != LJ_COMB_SINGLE
    int type_i = atom->type[i];
#endif
    for (int k = 0; k < nstencil; k++) {
        // __ldg: stencil/bins/x/type are read-only here, but the compiler cannot
        // prove they don't alias the neighbor-list writes, so it won't use the
        // read-only cache on its own (same as the force kernel)
        int jbin     = ibin + __ldg(&stencil[k]);
        int* loc_bin = &bins[jbin * atoms_per_bin];
        int bincnt   = __ldg(&bincount[jbin]);

        for (int m = 0; m < bincnt; m++) {
            int j = __ldg(&loc_bin[m]);
            if (j == i || (halfneigh && (j < i))) {
                continue;
            }

            MD_FLOAT delx = xtmp - __ldg(&atom_x(j));
            MD_FLOAT dely = ytmp - __ldg(&atom_y(j));
            MD_FLOAT delz = ztmp - __ldg(&atom_z(j));
            MD_FLOAT rsq  = delx * delx + dely * dely + delz * delz;

#if LJ_COMB_RULE != LJ_COMB_SINGLE
            int type_j            = __ldg(&atom->type[j]);
            const MD_FLOAT cutoff = __ldg(&atom->cutneighsq[type_i * ntypes + type_j]);
#else
            const MD_FLOAT cutoff = cutneighsq;
#endif

            if (rsq <= cutoff) {
                neighs(neighbor->neighbors, i, n, nlocal, maxneighs) = j;
                n++;
            }
        }
    }

    neighbor->numneigh[i] = n;
    if (n > maxneighs) {
        atomicMax(new_maxneighs, n);
    }
}

static Neighbor_params makeNeighborParams(void)
{
    Neighbor_params np { .xprd = xprd,
        .yprd                  = yprd,
        .zprd                  = zprd,
        .bininvx               = bininvx,
        .bininvy               = bininvy,
        .bininvz               = bininvz,
        .mbinxlo               = mbinxlo,
        .mbinylo               = mbinylo,
        .mbinzlo               = mbinzlo,
        .nbinx                 = nbinx,
        .nbiny                 = nbiny,
        .nbinz                 = nbinz,
        .mbinx                 = mbinx,
        .mbiny                 = mbiny,
        .mbinz                 = mbinz,
        // MultiGPU
        .pad_x    = pad_x,
        .pad_y    = pad_y,
        .pad_z    = pad_z,
        .binsizex = binsizex,
        .binsizey = binsizey,
        .binsizez = binsizez };
    return np;
}

/* gathers positions and velocities into bin order (one thread per bin,
 * preserving the deterministic in-bin ordering from sort_bin_contents_kernel,
 * so the permutation matches the host sortAtom exactly) */
__global__ void sort_gather_kernel(DeviceAtom a,
    MD_FLOAT* new_x,
    MD_FLOAT* new_vx,
    const int* binpos,
    const int* bincount,
    const int* bins,
    int atoms_per_bin,
    int mbins)
{
    const int mybin = blockIdx.x * blockDim.x + threadIdx.x;
    if (mybin >= mbins) {
        return;
    }

    DeviceAtom* atom = &a;
    const int start  = binpos[mybin];
    const int count  = bincount[mybin];
    for (int k = 0; k < count; k++) {
        const int ni       = start + k;
        const int oi       = bins[mybin * atoms_per_bin + k];
        new_x[ni * 3 + 0]  = atom_x(oi);
        new_x[ni * 3 + 1]  = atom_y(oi);
        new_x[ni * 3 + 2]  = atom_z(oi);
        new_vx[ni * 3 + 0] = atom->vx[oi * 3 + 0];
        new_vx[ni * 3 + 1] = atom->vx[oi * 3 + 1];
        new_vx[ni * 3 + 2] = atom->vx[oi * 3 + 2];
    }
}

void binatoms_cuda(Atom* atom,
    Binning* c_binning,
    int* c_resize_needed,
    Neighbor_params* np,
    const int threads_per_block)
{
    DEBUG_MESSAGE("binatoms_cuda begin\n");

    int nall             = atom->Nlocal + atom->Nghost;
    int resize           = 1;
    const int num_blocks = ceil((float)nall / (float)threads_per_block);

    while (resize > 0) {
        resize = 0;
        memsetGPU(c_binning->bincount, 0, c_binning->mbins * sizeof(int));
        memsetGPU(c_resize_needed, 0, sizeof(int));
        binatoms_kernel<<<num_blocks, threads_per_block>>>(atom->d_atom,
            atom->Nlocal + atom->Nghost,
            c_binning->bincount,
            c_binning->bins,
            c_binning->atoms_per_bin,
            *np,
            c_resize_needed);
        cuda_assert("binatoms", cudaPeekAtLastError());
        cuda_assert("binatoms", cudaDeviceSynchronize());
        memcpyFromGPU(&resize, c_resize_needed, sizeof(int));
        if (resize) {
            c_binning->atoms_per_bin *= 2;
            c_binning->bins = (int*)reallocateGPU(c_binning->bins,
                c_binning->mbins * c_binning->atoms_per_bin * sizeof(int));
        }
    }

    atoms_per_bin        = c_binning->atoms_per_bin;
    const int sortBlocks = ceil((float)mbins / (float)threads_per_block);
    sort_bin_contents_kernel<<<sortBlocks, threads_per_block>>>(c_binning->bincount,
        c_binning->bins,
        c_binning->mbins,
        c_binning->atoms_per_bin);
    cuda_assert("sort_bin", cudaPeekAtLastError());
    cuda_assert("sort_bin", cudaDeviceSynchronize());

    DEBUG_MESSAGE("binatoms_cuda end\n");
}

void buildNeighborCUDA(Atom* atom, Neighbor* neighbor)
{
    DEBUG_MESSAGE("buildNeighborCUDA begin\n");
    DeviceNeighbor* d_neighbor      = &(neighbor->d_neighbor);
    const int num_threads_per_block = get_cuda_num_threads();
    GPU_PROFILE_START("build_neighbor");

    int nall = atom->Nlocal + atom->Nghost;
    if (nall > nmax) {
        nmax                  = nall;
        d_neighbor->neighbors = (int*)reallocateGPU(d_neighbor->neighbors,
            nmax * neighbor->maxneighs * sizeof(int*));
        d_neighbor->numneigh  = (int*)reallocateGPU(d_neighbor->numneigh,
            nmax * sizeof(int));
    }

    // TODO move all of this initialization into its own method
    if (c_stencil == NULL) {
        c_stencil = (int*)allocateGPU(nstencil * sizeof(int));
        memcpyToGPU(c_stencil, stencil, nstencil * sizeof(int));
    }

    if (c_binning.mbins == 0) {
        c_binning.mbins         = mbins;
        c_binning.atoms_per_bin = atoms_per_bin;
        c_binning.bincount      = (int*)allocateGPU(c_binning.mbins * sizeof(int));
        c_binning.bins          = (int*)allocateGPU(
            c_binning.mbins * c_binning.atoms_per_bin * sizeof(int));
    }

    Neighbor_params np = makeNeighborParams();

    if (c_resize_needed == NULL) {
        c_resize_needed = (int*)allocateGPU(sizeof(int));
    }

    /* bin local & ghost atoms */
    binatoms_cuda(atom, &c_binning, c_resize_needed, &np, num_threads_per_block);
    if (c_new_maxneighs == NULL) {
        c_new_maxneighs = (int*)allocateGPU(sizeof(int));
    }

    int resize = 1;

    /* loop over each atom, storing neighbors */
    while (resize) {
        resize = 0;
        memsetGPU(c_new_maxneighs, 0, sizeof(int));
        const int num_blocks = ceil((float)atom->Nlocal / (float)num_threads_per_block);
        compute_neighborhood<<<num_blocks, num_threads_per_block>>>(atom->d_atom,
            *d_neighbor,
            np,
            atom->Nlocal,
            neighbor->maxneighs,
            neighbor->half_neigh,
            nstencil,
            c_stencil,
            c_binning.bins,
            c_binning.atoms_per_bin,
            c_binning.bincount,
            c_new_maxneighs,
            cutneighsq,
            atom->ntypes);

        cuda_assert("compute_neighborhood", cudaPeekAtLastError());
        cuda_assert("compute_neighborhood", cudaDeviceSynchronize());

        int new_maxneighs;
        memcpyFromGPU(&new_maxneighs, c_new_maxneighs, sizeof(int));
        if (new_maxneighs > neighbor->maxneighs) {
            resize = 1;
        }

        if (resize) {
            printf("RESIZE %d\n", neighbor->maxneighs);
            neighbor->maxneighs = new_maxneighs * 1.2;
            printf("NEW SIZE %d\n", neighbor->maxneighs);
            neighbor->neighbors = (int*)reallocateGPU(neighbor->neighbors,
                atom->Nmax * neighbor->maxneighs * sizeof(int));
        }
    }

    GPU_PROFILE_STOP();
    DEBUG_MESSAGE("buildNeighborCUDA end\n");
}

/* device-resident sortAtom: same bin-order permutation as the host sortAtom
 * (neighbor.c), but positions/velocities never leave the device. Caller has
 * set Nghost = 0; neighbor list and ghosts are rebuilt in the same step.
 * ponytail: type/epsilon/sigma6 not permuted, matches host sortAtom; three
 * more gather lines if multi-type + sort is ever benchmarked */
void sortAtomCUDA(Atom* atom)
{
    DEBUG_MESSAGE("sortAtomCUDA begin\n");
    const int num_threads_per_block = get_cuda_num_threads();

    // defensive init: in practice buildNeighborCUDA has run during setup
    if (c_binning.mbins == 0) {
        c_binning.mbins         = mbins;
        c_binning.atoms_per_bin = atoms_per_bin;
        c_binning.bincount      = (int*)allocateGPU(c_binning.mbins * sizeof(int));
        c_binning.bins          = (int*)allocateGPU(
            c_binning.mbins * c_binning.atoms_per_bin * sizeof(int));
    }

    if (c_resize_needed == NULL) {
        c_resize_needed = (int*)allocateGPU(sizeof(int));
    }

    Neighbor_params np = makeNeighborParams();

    /* bins local atoms only (Nghost == 0) and sorts each bin's contents */
    binatoms_cuda(atom, &c_binning, c_resize_needed, &np, num_threads_per_block);

    if (c_binpos == NULL) {
        c_binpos = (int*)allocateGPU(c_binning.mbins * sizeof(int));
    }

    // exclusive scan: bincount -> per-bin write offsets
    size_t tempBytes = 0;
    cuda_assert("sortAtom.scanSize",
        cub::DeviceScan::ExclusiveSum(NULL,
            tempBytes,
            c_binning.bincount,
            c_binpos,
            c_binning.mbins));
    if (tempBytes > c_sortScanTempCap) {
        c_sortScanTempCap = tempBytes;
        c_sortScanTemp    = reallocateGPU(c_sortScanTemp, tempBytes);
    }
    cuda_assert("sortAtom.scan",
        cub::DeviceScan::ExclusiveSum(c_sortScanTemp,
            tempBytes,
            c_binning.bincount,
            c_binpos,
            c_binning.mbins));

    // scratch fully rewritten below (up to Nlocal; the rest is rebuilt as
    // ghosts before any read), so plain reallocateGPU suffices
    if (c_sortNmax < atom->Nmax) {
        c_sortNmax = atom->Nmax;
        c_sort_x  = (MD_FLOAT*)reallocateGPU(c_sort_x, c_sortNmax * sizeof(MD_FLOAT) * 3);
        c_sort_vx = (MD_FLOAT*)reallocateGPU(c_sort_vx,
            c_sortNmax * sizeof(MD_FLOAT) * 3);
    }

    const int num_blocks = ceil((float)c_binning.mbins / (float)num_threads_per_block);
    sort_gather_kernel<<<num_blocks, num_threads_per_block>>>(atom->d_atom,
        c_sort_x,
        c_sort_vx,
        c_binpos,
        c_binning.bincount,
        c_binning.bins,
        c_binning.atoms_per_bin,
        c_binning.mbins);
    cuda_assert("sort_gather", cudaPeekAtLastError());
    cuda_assert("sort_gather", cudaDeviceSynchronize());

    // swap sorted buffers in; both are Nmax*3-sized, so a later growAtom
    // (reallocateGPUKeep) stays correct
    MD_FLOAT* tmp   = atom->d_atom.x;
    atom->d_atom.x  = c_sort_x;
    c_sort_x        = tmp;
    tmp             = atom->d_atom.vx;
    atom->d_atom.vx = c_sort_vx;
    c_sort_vx       = tmp;

    DEBUG_MESSAGE("sortAtomCUDA end\n");
}
