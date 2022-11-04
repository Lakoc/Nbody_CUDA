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
 * Aux function to get size of dynamic shared memory
 * Author: einpoklum  Source: https://stackoverflow.com/questions/42309369/can-my-kernel-code-tell-how-much-shared-memory-it-has-available
 * @return Size of dynamic shared memory
 */
__forceinline__ __device__ unsigned dynamic_smem_size() {
    unsigned ret;
    asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}

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
    extern __shared__ float shared[];
    int elements_to_cache = (int) (dynamic_smem_size() / (blockDim.x * sizeof(float)));

    unsigned global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id >= N) return;

    float pos_x = p_curr.pos_x[global_id];
    float pos_y = p_curr.pos_y[global_id];
    float pos_z = p_curr.pos_z[global_id];
    float p1_weight = p_curr.weight[global_id];
    float vel_x = p_curr.vel_x[global_id];
    float vel_y = p_curr.vel_y[global_id];
    float vel_z = p_curr.vel_z[global_id];

    bool in_mem_pos_x = elements_to_cache <= POS_X;
    bool in_mem_pos_y = elements_to_cache <= POS_Y;
    bool in_mem_pos_z = elements_to_cache <= POS_Z;
    bool in_mem_vel_x = elements_to_cache <= VEL_X;
    bool in_mem_vel_y = elements_to_cache <= VEL_Y;
    bool in_mem_vel_z = elements_to_cache <= VEL_Z;
    bool in_mem_weight = elements_to_cache <= WEIGHT;

    float *mem_pos_x = !in_mem_pos_x ? &shared[POS_X * blockDim.x] : p_curr.pos_x;
    float *mem_pos_y = !in_mem_pos_y ? &shared[POS_Y * blockDim.x] : p_curr.pos_y;
    float *mem_pos_z = !in_mem_pos_z ? &shared[POS_Z * blockDim.x] : p_curr.pos_z;
    float *mem_vel_x = !in_mem_vel_x ? &shared[VEL_X * blockDim.x] : p_curr.vel_x;
    float *mem_vel_y = !in_mem_vel_y ? &shared[VEL_Y * blockDim.x] : p_curr.vel_y;
    float *mem_vel_z = !in_mem_vel_z ? &shared[VEL_Z * blockDim.x] : p_curr.vel_z;
    float *mem_weight = !in_mem_weight ? &shared[WEIGHT * blockDim.x] : p_curr.weight;

    float *global_arrays[N_ELEMENTS] = {p_curr.pos_x, p_curr.pos_y, p_curr.pos_z, p_curr.vel_x, p_curr.vel_y,
                                        p_curr.vel_z, p_curr.weight};

    float r, dx, dy, dz, r3, Fg_dt_m2_r, p2_weight, weight_difference, weight_sum, double_m2;
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    bool colliding;
    unsigned load_index;
    unsigned block_offset;

    for (int tile = 0; tile < gridDim.x; tile++) {
        block_offset = tile * blockDim.x;
        load_index = block_offset + threadIdx.x;

        for (int j = 0; j < elements_to_cache; j++) {
            shared[j * blockDim.x + threadIdx.x] = global_arrays[j][load_index];
        }
        __syncthreads();

        for (int i = 0; i < blockDim.x; i++) {

            dx = mem_pos_x[block_offset * in_mem_pos_x + i] - pos_x;
            dy = mem_pos_y[block_offset * in_mem_pos_y + i] - pos_y;
            dz = mem_pos_z[block_offset * in_mem_pos_z + i] - pos_z;

            r = sqrt(dx * dx + dy * dy + dz * dz);
            r3 = r * r * r + FLT_MIN;
            colliding = r > 0.0f && r <= COLLISION_DISTANCE;


            p2_weight = block_offset + i < N ? mem_weight[block_offset * in_mem_weight + i] : 0.0f;
            weight_difference = p1_weight - p2_weight;
            weight_sum = p1_weight + p2_weight;
            double_m2 = p2_weight * 2.0f;

            Fg_dt_m2_r = G * dt / r3 * p2_weight;


            vx += colliding ?
                  ((vel_x * weight_difference + double_m2 * mem_vel_x[block_offset * in_mem_vel_x + i]) / weight_sum) -
                  vel_x :
                  Fg_dt_m2_r * dx;
            vy += colliding ?
                  ((vel_y * weight_difference + double_m2 * mem_vel_y[block_offset * in_mem_vel_y + i]) / weight_sum) -
                  vel_y :
                  Fg_dt_m2_r * dy;
            vz += colliding ?
                  ((vel_z * weight_difference + double_m2 * mem_vel_z[block_offset * in_mem_vel_z + i]) / weight_sum) -
                  vel_z :
                  Fg_dt_m2_r * dz;
        }
        __syncthreads();

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
