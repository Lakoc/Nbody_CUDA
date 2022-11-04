/**
 * @File nbody.cu
 *
 * Implementation of the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2022)
 * Projekt c. 1 (cuda)
 * Login: xpolok03
 */

#include <cmath>
#include <cfloat>
#include "nbody.h"

/**
 * CUDA kernel to calculate gravitation velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_velocity(t_particles p_curr,
                                   t_particles p_next,
                                   int N,
                                   float dt) {
    unsigned global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id >= N){
        return;
    }
    float pos_x = p_curr.pos_x[global_id];
    float pos_y = p_curr.pos_y[global_id];
    float pos_z = p_curr.pos_z[global_id];
    float p1_weight = p_curr.weight[global_id];
    float vel_x = p_curr.vel_x[global_id];
    float vel_y = p_curr.vel_y[global_id];
    float vel_z = p_curr.vel_z[global_id];


    float r, dx, dy, dz, r3, Fg_dt_m2_r, p2_weight, weight_difference, weight_sum, double_m2;
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    bool colliding;

    for (int i = 0; i < N; i++) {
        dx = p_curr.pos_x[i] - pos_x;
        dy = p_curr.pos_y[i] - pos_y;
        dz = p_curr.pos_z[i] - pos_z;
        r = sqrt(dx * dx + dy * dy + dz * dz);
        r3 = r * r * r + FLT_MIN;
        colliding = r > 0.0f && r <= COLLISION_DISTANCE;


        p2_weight = p_curr.weight[i];
        weight_difference = p1_weight - p2_weight;
        weight_sum = p1_weight + p2_weight;
        double_m2 = p2_weight * 2.0f;

        Fg_dt_m2_r = G * dt / r3 * p2_weight;


        vx += colliding ? ((vel_x * weight_difference + double_m2 * p_curr.vel_x[i]) / weight_sum) - vel_x : Fg_dt_m2_r * dx;
        vy += colliding ? ((vel_y * weight_difference + double_m2 * p_curr.vel_y[i]) / weight_sum) - vel_y : Fg_dt_m2_r * dy;
        vz += colliding ? ((vel_z * weight_difference + double_m2 * p_curr.vel_z[i]) / weight_sum) - vel_z : Fg_dt_m2_r * dz;
    }

    vel_x += vx;
    vel_y += vy;
    vel_z += vz;

    p_next.vel_x[global_id] = vel_x;
    p_next.vel_y[global_id] = vel_y;
    p_next.vel_z[global_id] = vel_z;

    p_next.pos_x[global_id] = pos_x + vel_x * dt;
    p_next.pos_y[global_id] = pos_y + vel_y * dt;
    p_next.pos_z[global_id] = pos_z + vel_z * dt;

}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------


__global__ void
centerOfMass(t_particles p, float *comX, float *comY, float *comZ, float *comW, int *lock, const int N) {

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassCPU(MemDesc &memDesc) {
    float4 com = {0, 0, 0, 0};

    for (int i = 0; i < memDesc.getDataSize(); i++) {
        // Calculate the vector on the line connecting current body and most recent position of center-of-mass
        const float dx = memDesc.getPosX(i) - com.x;
        const float dy = memDesc.getPosY(i) - com.y;
        const float dz = memDesc.getPosZ(i) - com.z;

        // Calculate weight ratio only if at least one particle isn't massless
        const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                         ? (memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

        // Update position and weight of the center-of-mass according to the weight ration and vector
        com.x += dx * dw;
        com.y += dy * dw;
        com.z += dz * dw;
        com.w += memDesc.getWeight(i);
    }
    return com;
}// enf of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
