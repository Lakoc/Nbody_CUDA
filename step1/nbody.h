/**
 * @File nbody.h
 *
 * Header file of your implementation to the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2022)
 * Projekt c. 1 (cuda)
 * Login: xpolok03
 */

#ifndef __NBODY_H__
#define __NBODY_H__

#include <cstdlib>
#include <cstdio>
#include "h5Helper.h"

/* Gravitation constant */
constexpr float G = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;


/**
 * Particles data structure
 */
typedef struct {
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                 FILL IN: Particle data structure optimal for the use on GPU (step 0)                             //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float4 *pos;
    float3 *vel;
} t_particles;


/**
 * CUDA kernel to calculate velocities
 * @param p_new       - structure to save new particle states
 * @param tmp_vel - structure holding current particle states
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_velocity(t_particles p_curr,
                                   t_particles p_next,
                                   int N,
                                   float dt);


__global__ void centerOfMass(t_particles p,
                             float *comX,
                             float *comY,
                             float *comZ,
                             float *comW,
                             int *lock,
                             const int N);

/**
 * CPU implementation of the Center of Mass calculation
 * @param memDesc - Memory descriptor of particle data on CPU side
 */
float4 centerOfMassCPU(MemDesc &memDesc);

#endif /* __NBODY_H__ */
