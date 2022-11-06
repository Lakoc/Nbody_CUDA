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
    // If out of bounds, there is no work -> small divergence at last block
    if (global_id >= N) {
        return;
    }

    // Auxiliary registers to store p1 and p2 current positions, weights and velocities
    float4 pos_p1 = p_curr.pos[global_id];
    float3 vel_p1 = p_curr.vel[global_id];
    float4 pos_p2;
    float3 vel_p2;

    // Aux variables used in loop below
    float r, dx, dy, dz, r3, Fg_dt_m2_r, weight_difference, weight_sum, double_m2;

    // Temp vector to store collision velocities sum, no need to access global memory at each iteration
    float3 v_temp = {0.0f, 0.0f, 0.0f};
    bool colliding;

    for (int i = 0; i < N; i++) {
        // Load particle_2 data
        pos_p2 = p_curr.pos[i];
        vel_p2 = p_curr.vel[i];

        // Calculate per axis distance
        // In case of collision, distance are only used to calculate Euclidean distance,
        // thus they can be kept in reversed order
        dx = pos_p2.x - pos_p1.x;
        dy = pos_p2.y - pos_p1.y;
        dz = pos_p2.z - pos_p1.z;

        // Calculate Euclidean distance between two particles
        r = sqrt(dx * dx + dy * dy + dz * dz);
        r3 = r * r * r + FLT_MIN;

        colliding = r > 0.0f && r <= COLLISION_DISTANCE;

        // Save values below to registers to save accesses to memory and multiple calculations of same code
        weight_difference = pos_p1.w - pos_p2.w;
        weight_sum = pos_p1.w + pos_p2.w;
        double_m2 = pos_p2.w * 2.0f;
        Fg_dt_m2_r = G * dt / r3 * pos_p2.w;

        // If there is collision add collision velocities, otherwise gravitational ->
        // gravitational velocities are skipped if there is collision, likewise vice versa
        v_temp.x += colliding ? ((vel_p1.x * weight_difference + double_m2 * vel_p2.x) / weight_sum) - vel_p1.x :
                    Fg_dt_m2_r * dx;
        v_temp.y += colliding ? ((vel_p1.y * weight_difference + double_m2 * vel_p2.y) / weight_sum) - vel_p1.y :
                    Fg_dt_m2_r * dy;
        v_temp.z += colliding ? ((vel_p1.z * weight_difference + double_m2 * vel_p2.z) / weight_sum) - vel_p1.z :
                    Fg_dt_m2_r * dz;
    }

    // Update values in global context
    vel_p1.x += v_temp.x;
    vel_p1.y += v_temp.y;
    vel_p1.z += v_temp.z;
    p_next.vel[global_id] = vel_p1;

    // Positions are updated with current velocities
    pos_p1.x += vel_p1.x * dt;
    pos_p1.y += vel_p1.y * dt;
    pos_p1.z += vel_p1.z * dt;
    p_next.pos[global_id] = pos_p1;


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
