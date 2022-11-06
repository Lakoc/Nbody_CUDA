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
__global__ void calculate_gravitation_velocity(t_particles p, t_velocities tmp_vel, int N, float dt) {
    unsigned global_id = threadIdx.x + blockIdx.x * blockDim.x;
    // If out of bounds, there is no work -> small divergence at last block
    if (global_id >= N){
        return;
    }

    // Load particle_1 positions and weight
    float4 pos_p1 = p.pos[global_id];
    // Declare vector to hold particle_2 data
    float4 pos_p2;

    // Aux variables used in loop below
    float r, dx, dy, dz, r3, Fg_dt_m2_r;;

    // Temp vector to store gravitation velocities sum, no need to access global memory at each iteration
    float3 v_temp = {0.0f, 0.0f, 0.0f};

    bool not_colliding;

    for (int i = 0; i < N; i++) {
        // Load particle2 to local registers
        pos_p2 = p.pos[i];

        // Calculate per axis distance
        // Reverted order to save up 1 more unary operation (-G  -> G)
        dx = pos_p2.x - pos_p1.x;
        dy = pos_p2.y - pos_p1.y;
        dz =pos_p2.z - pos_p1.z;

        // Calculate Euclidean distance between two particles
        r = sqrt(dx * dx + dy * dy + dz * dz);
        r3 = r * r * r + FLT_MIN;

        // Simplified from CPU implementation
        Fg_dt_m2_r = G * dt / r3 * pos_p2.w;

        not_colliding = r > COLLISION_DISTANCE;

        // If there is no collision, add local velocities to temporal vector
        v_temp.x += not_colliding ? Fg_dt_m2_r * dx : 0.0f;
        v_temp.y += not_colliding ? Fg_dt_m2_r * dy : 0.0f;
        v_temp.z += not_colliding ? Fg_dt_m2_r * dz : 0.0f;
    }

    // Update values in global context
    tmp_vel.vel[global_id].x = v_temp.x;
    tmp_vel.vel[global_id].y = v_temp.y;
    tmp_vel.vel[global_id].z = v_temp.z;

}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_collision_velocity(t_particles p, t_velocities tmp_vel, int N, float dt) {
    unsigned global_id = threadIdx.x + blockIdx.x * blockDim.x;

    // If out of bounds, there is no work -> small divergence at last block
    if (global_id > N){
        return;
    }

    // Auxiliary registers to store p1 and p2 current positions, weights and velocities
    float4 pos_p1 = p.pos[global_id];
    float3 vel_p1 = p.vel[global_id];
    float4 pos_p2;
    float3 vel_p2;

    // Aux variables used in loop below
    float r, dx, dy, dz, weight_difference, weight_sum, double_m2;

    // Temp vector to store collision velocities sum, no need to access global memory at each iteration
    float3 v_temp = {0.0f, 0.0f, 0.0f};

    bool colliding;

    for (int i = 0; i < N; i++) {
        // Load particle_2 data
        pos_p2 = p.pos[i];
        vel_p2 = p.vel[i];

        // Calculate per axis distance
        dx = pos_p1.x - pos_p2.x;
        dy = pos_p1.y - pos_p2.y;
        dz = pos_p1.z - pos_p2.z;

        // Calculate Euclidean distance between two particles
        r = sqrt(dx * dx + dy * dy + dz * dz);

        // Save values below to registers to save accesses to memory and multiple calculations of same code
        weight_difference = pos_p1.w - pos_p2.w;
        weight_sum = pos_p1.w + pos_p2.w;
        double_m2 = pos_p2.w * 2.0f;

        colliding = r > 0.0f && r <= COLLISION_DISTANCE;

        // If colliding add to temporal vector current velocities
        // Application of distributive law of *,+ operations in Real field => p1.weight* p1.vel_x - p2.weight *p1.vel_x  - > p1.vel_x * (weight_difference)
        v_temp.x += colliding ? ((vel_p1.x * weight_difference + double_m2 * vel_p2.x) / weight_sum) - vel_p1.x : 0.0f;
        v_temp.y += colliding ? ((vel_p1.y * weight_difference + double_m2 * vel_p2.y) / weight_sum) - vel_p1.y : 0.0f;
        v_temp.z += colliding ? ((vel_p1.z * weight_difference + double_m2 * vel_p2.z) / weight_sum) - vel_p1.z : 0.0f;
    }

    // Update values in global context
    tmp_vel.vel[global_id].x += v_temp.x;
    tmp_vel.vel[global_id].y += v_temp.y;
    tmp_vel.vel[global_id].z += v_temp.z;
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void update_particle(t_particles p, t_velocities tmp_vel, int N, float dt) {
    unsigned global_id = threadIdx.x + blockIdx.x * blockDim.x;
    // small divergence to ensure there is no segfault
    if (global_id > N){
        return;
    }
    p.vel[global_id].x += tmp_vel.vel[global_id].x;
    p.vel[global_id].y += tmp_vel.vel[global_id].y;
    p.vel[global_id].z += tmp_vel.vel[global_id].z;
    p.pos[global_id].x += p.vel[global_id].x * dt;
    p.pos[global_id].y += p.vel[global_id].y * dt;
    p.pos[global_id].z += p.vel[global_id].z * dt;
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param comX    - pointer to a center of mass position in X
 * @param comY    - pointer to a center of mass position in Y
 * @param comZ    - pointer to a center of mass position in Z
 * @param comW    - pointer to a center of mass weight
 * @param lock    - pointer to a user-implemented lock
 * @param N       - Number of particles
 */
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
