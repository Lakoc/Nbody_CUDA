/**
 * @File main.cu
 *
 * The main file of the project
 *
 * Paralelní programování na GPU (PCG 2022)
 * Projekt c. 1 (cuda)
 * Login: xpolok03
 */

#include <sys/time.h>
#include <cstdio>
#include <cmath>

#include "nbody.h"
#include "h5Helper.h"

/**
 * Aux macro and function to log last cuda err
 * Author: Robert Crovella, talonmies  Source: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {
    // Time measurement
    struct timeval t1, t2;

    if (argc != 10) {
        printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
        exit(1);
    }

    // Number of particles
    const int N = std::stoi(argv[1]);
    // Length of time step
    const float dt = std::stof(argv[2]);
    // Number of steps
    const int steps = std::stoi(argv[3]);
    // Number of thread blocks
    const int thr_blc = std::stoi(argv[4]);
    // Write frequency
    int writeFreq = std::stoi(argv[5]);
    // number of reduction threads
    const int red_thr = std::stoi(argv[6]);
    // Number of reduction threads/blocks
    const int red_thr_blc = std::stoi(argv[7]);

    // Size of the simulation CUDA gird - number of blocks
    const size_t simulationGrid = (N + thr_blc - 1) / thr_blc;
    // Size of the reduction CUDA grid - number of blocks
    const size_t reductionGrid = (red_thr + red_thr_blc - 1) / red_thr_blc;

    // Log benchmark setup
    printf("N: %d\n", N);
    printf("dt: %f\n", dt);
    printf("steps: %d\n", steps);
    printf("threads/block: %d\n", thr_blc);
    printf("blocks/grid: %lu\n", simulationGrid);
    printf("reduction threads/block: %d\n", red_thr_blc);
    printf("reduction blocks/grid: %lu\n", reductionGrid);

    const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;
    writeFreq = (writeFreq > 0) ? writeFreq : 0;


    t_particles particles_cpu;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                            FILL IN: CPU side memory allocation (step 0)                                          //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    size_t particles_pos_arr_size = N * sizeof(float4);
    size_t particles_vel_arr_size = N * sizeof(float3);

    particles_cpu.pos = static_cast<float4 *>(malloc(particles_pos_arr_size));
    particles_cpu.vel = static_cast<float3 *>(malloc(particles_vel_arr_size));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                              FILL IN: memory layout descriptor (step 0)                                          //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
     * Caution! Create only after CPU side allocation
     * parameters:
     *                      Stride of two               Offset of the first
     *  Data pointer        consecutive elements        element in floats,
     *                      in floats, not bytes        not bytes
    */
    MemDesc md(
            &particles_cpu.pos->x, 4, 0,              // Postition in X
            &particles_cpu.pos->y, 4, 0,              // Postition in Y
            &particles_cpu.pos->z, 4, 0,              // Postition in Z
            &particles_cpu.vel->x, 3, 0,              // Velocity in X
            &particles_cpu.vel->y, 3, 0,              // Velocity in Y
            &particles_cpu.vel->z, 3, 0,              // Velocity in Z
            &particles_cpu.pos->w, 4, 0,              // Weight
            N,                                                                  // Number of particles
            recordsNum);                                                             // Number of records in output file

    // Initialisation of helper class and loading of input data
    H5Helper h5Helper(argv[8], argv[9], md);

    try {
        h5Helper.init();
        h5Helper.readParticleData();
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }


    t_particles particles_gpu_curr;
    t_particles particles_gpu_next;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                  FILL IN: GPU side memory allocation (step 0)                                    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    gpuErrCheck(cudaMalloc<float4>(&particles_gpu_curr.pos, particles_pos_arr_size))
    gpuErrCheck(cudaMalloc<float3>(&particles_gpu_curr.vel, particles_vel_arr_size))

    gpuErrCheck(cudaMalloc<float4>(&particles_gpu_next.pos, particles_pos_arr_size))
    gpuErrCheck(cudaMalloc<float3>(&particles_gpu_next.vel, particles_vel_arr_size))

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       FILL IN: memory transfers (step 0)                                         //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    gpuErrCheck(cudaMemcpy(particles_gpu_curr.pos, particles_cpu.pos, particles_pos_arr_size, cudaMemcpyHostToDevice))
    gpuErrCheck(cudaMemcpy(particles_gpu_curr.vel, particles_cpu.vel, particles_vel_arr_size, cudaMemcpyHostToDevice))

    // There is no need to copy velocities to next state of particles, they will never be accessed until calculation of next step
    gpuErrCheck(cudaMemcpy(particles_gpu_next.pos, particles_cpu.pos, particles_vel_arr_size, cudaMemcpyHostToDevice))


    dim3 dimBlock(thr_blc);
    dim3 dimGrid(simulationGrid);

    gettimeofday(&t1, 0);

    for (int s = 0; s < steps; s++) {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                       FILL IN: kernels invocation (step 0)                                     //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        calculate_velocity<<<dimGrid, dimBlock>>>(particles_gpu_curr, particles_gpu_next, N, dt);
        std::swap(particles_gpu_curr, particles_gpu_next);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                          FILL IN: synchronization  (step 4)                                    //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (writeFreq > 0 && (s % writeFreq == 0)) {
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                          FILL IN: synchronization and file access logic (step 4)                             //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //              FILL IN: invocation of center-of-mass kernel (step 3.1, step 3.2, step 4)                           //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    gpuErrCheck(cudaDeviceSynchronize())
    gpuErrCheck(cudaPeekAtLastError())

    gettimeofday(&t2, 0);

    // Approximate simulation wall time
    double t = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("Time: %f s\n", t);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                             FILL IN: memory transfers for particle data (step 0)                                 //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float4 comOnGPU;

    cudaMemcpy(particles_cpu.pos, particles_gpu_curr.pos, particles_pos_arr_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles_cpu.vel, particles_gpu_curr.vel, particles_vel_arr_size, cudaMemcpyDeviceToHost);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                        FILL IN: memory transfers for center-of-mass (step 3.1, step 3.2)                         //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float4 comOnCPU = centerOfMassCPU(md);

    std::cout << "Center of mass on CPU:" << std::endl
              << comOnCPU.x << ", "
              << comOnCPU.y << ", "
              << comOnCPU.z << ", "
              << comOnCPU.w
              << std::endl;

    std::cout << "Center of mass on GPU:" << std::endl
              << comOnGPU.x << ", "
              << comOnGPU.y << ", "
              << comOnGPU.z << ", "
              << comOnGPU.w
              << std::endl;

    // Writing final values to the file
    h5Helper.writeComFinal(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w);
    h5Helper.writeParticleDataFinal();

    // Memory cleanup
    free(particles_cpu.pos);
    free(particles_cpu.vel);

    cudaFree(particles_gpu_curr.pos);
    cudaFree(particles_gpu_curr.vel);
    cudaFree(particles_gpu_next.pos);
    cudaFree(particles_gpu_next.vel);



    return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
