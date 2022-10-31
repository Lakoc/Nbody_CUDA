/**
 * @File main.cu
 *
 * The main file of the project
 *
 * Paralelní programování na GPU (PCG 2022)
 * Projekt c. 1 (cuda)
 * Login: xlogin00
 */

#include <sys/time.h>
#include <cmath>
#include <cstdio>
#include <cmath>

#include "nbody.h"
#include "h5Helper.h"

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
    size_t particle_arr_size = N * sizeof(float);
    particles_cpu.pos_x = static_cast<float *>(malloc(particle_arr_size));
    particles_cpu.pos_y = static_cast<float *>(malloc(particle_arr_size));
    particles_cpu.pos_z = static_cast<float *>(malloc(particle_arr_size));
    particles_cpu.vel_x = static_cast<float *>(malloc(particle_arr_size));
    particles_cpu.vel_y = static_cast<float *>(malloc(particle_arr_size));
    particles_cpu.vel_z = static_cast<float *>(malloc(particle_arr_size));
    particles_cpu.vel_z = static_cast<float *>(malloc(particle_arr_size));

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
            particles_cpu.pos_x, 1, 0,              // Postition in X
            particles_cpu.pos_y, 1, 0,              // Postition in Y
            particles_cpu.pos_z, 1, 0,              // Postition in Z
            particles_cpu.vel_x, 1, 0,              // Velocity in X
            particles_cpu.vel_y, 1, 0,              // Velocity in Y
            particles_cpu.vel_z, 1, 0,              // Velocity in Z
            particles_cpu.weight, 1, 0,              // Weight
            N,                                                                  // Number of particles
            recordsNum);                                                        // Number of records in output file

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


    t_particles particles_gpu;
    t_velocities velocities_gpu;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                  FILL IN: GPU side memory allocation (step 0)                                    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaMalloc<float>(&particles_gpu.pos_x, particle_arr_size);
    cudaMalloc<float>(&particles_gpu.pos_y, particle_arr_size);
    cudaMalloc<float>(&particles_gpu.pos_z, particle_arr_size);
    cudaMalloc<float>(&particles_gpu.vel_x, particle_arr_size);
    cudaMalloc<float>(&particles_gpu.vel_y, particle_arr_size);
    cudaMalloc<float>(&particles_gpu.vel_z, particle_arr_size);
    cudaMalloc<float>(&particles_gpu.weight, particle_arr_size);

    cudaMalloc<float>(&velocities_gpu.x, particle_arr_size);
    cudaMalloc<float>(&velocities_gpu.y, particle_arr_size);
    cudaMalloc<float>(&velocities_gpu.z, particle_arr_size);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       FILL IN: memory transfers (step 0)                                         //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaMemcpy(particles_gpu.pos_x, particles_cpu.pos_x, particle_arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(particles_gpu.pos_y, particles_cpu.pos_y, particle_arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(particles_gpu.pos_z, particles_cpu.pos_z, particle_arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(particles_gpu.vel_x, particles_cpu.vel_x, particle_arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(particles_gpu.vel_y, particles_cpu.vel_y, particle_arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(particles_gpu.vel_z, particles_cpu.vel_z, particle_arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(particles_gpu.weight, particles_cpu.weight, particle_arr_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(thr_blc);

    dim3 dimGrid(std::ceil(float(N) / float(thr_blc)));

    gettimeofday(&t1, 0);

    for (int s = 0; s < steps; s++) {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                       FILL IN: kernels invocation (step 0)                                     //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        calculate_gravitation_velocity<<<dimGrid, dimBlock>>>(particles_gpu, velocities_gpu, N,dt);
        calculate_collision_velocity<<<dimGrid, dimBlock>>>(particles_gpu, velocities_gpu, N,dt);
        update_particle<<<dimGrid, dimBlock>>>(particles_gpu, velocities_gpu, N,dt);


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
    cudaDeviceSynchronize();

    gettimeofday(&t2, 0);

    // Approximate simulation wall time
    double t = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("Time: %f s\n", t);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                             FILL IN: memory transfers for particle data (step 0)                                 //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float4 comOnGPU;
    cudaMemcpy(particles_cpu.pos_x, particles_gpu.pos_x, particle_arr_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles_cpu.pos_y, particles_gpu.pos_y, particle_arr_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles_cpu.pos_z, particles_gpu.pos_z, particle_arr_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles_cpu.vel_x, particles_gpu.vel_x, particle_arr_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles_cpu.vel_y, particles_gpu.vel_y, particle_arr_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles_cpu.vel_z, particles_gpu.vel_z, particle_arr_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles_cpu.weight, particles_gpu.weight, particle_arr_size, cudaMemcpyDeviceToHost);


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

    return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
