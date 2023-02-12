/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER ORasdadsa
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <helper_cuda.h>
#include <chrono>
#include <vector>
#include <random>


//simple vector class with operator overloading for adding vectors and a randomized initialization:

class Vector {
public:
    Vector(int n) : data(n) {
        // Randomly initialize the vector with values between 0 and 100
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, 100);

        for (int i = 0; i < n; i++) {
            data[i] = dist(gen);
        }
    }

    Vector operator+(const Vector& other) const {
        Vector result(data.size());
        for (int i = 0; i < data.size(); i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector& vec) {
        os << "[";
        for (int i = 0; i < vec.data.size(); i++) {
            os << vec.data[i];
            if (i != vec.data.size() - 1) {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }

private:
    std::vector<int> data;
};

/////////////////////////////////////////////////////////////////
// Some utility code to define grid_stride_range
// Normally this would be in a header but it's here
// for didactic purposes. Uses
#include "range.hpp"
using namespace util::lang;

// type alias to simplify typing...
template <typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

//compiler decides whether or not to actually inline the function based on a variety of factors such as code size and the optimization level

template <typename T>
inline __device__ step_range<T> grid_stride_range(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x;
  return range(begin, end).step(gridDim.x * blockDim.x);
}

//simple operator overload - it allows you to add an integer to a grid_stride_range object 
//and return a new grid_stride_range object with its begin value increased by the integer value

//it's not actually used in any meaningful way in this code, it's just a representation of operator overloading
//lhs - left hand side, rhs - right hand side
template <typename T>
inline __device__ step_range<T> operator+(const step_range<T> &lhs, const int &rhs) {
  return range(lhs.begin() + rhs, lhs.end()).step(lhs.step());
}
/////////////////////////////////////////////////////////////////

// Overloading function count_if to take either a functor or a char value
// Simple SFINAE implementation - this function is enabled only when the type of Predicate is not char
template <typename T, typename Predicate,
			typename = typename std::enable_if<!std::is_same<Predicate, char>::value>::type>
__device__ void count_if(int *count, T *data, int n, Predicate p) {
  for (auto i : grid_stride_range(0, n)) {
    if (p(data[i])) atomicAdd(count, 1);
  }
}

template <typename T>
__device__ void count_if(int *count, T *data, int n, char value) {
  for (auto i : grid_stride_range(0, n)) {
    if (data[i] == value) atomicAdd(count, 1);
  }
}

// Use count_if with a lambda function that searches for x, y, z or w
// Note the use of range-based for loop and initializer_list inside the functor
// We use auto so we don't have to know the type of the functor or array
inline __global__ void xyzw_frequency(int *count, char *text, int n) {
  const char letters[]{'x', 'y', 'z', 'w'};

  count_if(count, text, n, [&](char c) {
    for (const auto x : letters)
      if (c == x) return true;
    return false;
  });
}

inline __global__ void xyzw_frequency_thrust_device(int *count, char *text, int n) {
  const char letters[]{'x', 'y', 'z', 'w'};
  *count = thrust::count_if(thrust::device, text, text + n, [=](char c) {
    for (const auto x : letters)
      if (c == x) return true;
    return false;
  });
}

// a bug in Thrust 1.8 causes warnings when this is uncommented
// so commented out by default -- fixed in Thrust master branch
#if 0 
void xyzw_frequency_thrust_host(int *count, char *text, int n)
{
  const char letters[] {'x', 'y', 'z', 'w'};
  *count = thrust::count_if(thrust::host, text, text+n, [&](char c) {
    for (const auto x : letters) 
      if (c == x) return true;
    return false;
  });
}
#endif

int main(int argc, char **argv) {
	
  std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();

  const char *filename = sdkFindFilePath("quovadis.txt", argv[0]);
  
  Vector v1(5), v2(5);

  int numBytes = 16 * 1048576;
  char *h_text = (char *)malloc(numBytes);

  // find first CUDA device
  int devID = findCudaDevice(argc, (const char **)argv);

  char *d_text;
  checkCudaErrors(cudaMalloc((void **)&d_text, numBytes));

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Cannot find the input text file\n. Exiting..\n");
    return EXIT_FAILURE;
  }
  int len = (int)fread(h_text, sizeof(char), numBytes, fp);
  fclose(fp);
  std::cout << "Read " << len << " byte corpus from " << filename << std::endl;

  checkCudaErrors(cudaMemcpy(d_text, h_text, len, cudaMemcpyHostToDevice));

  int count = 0;
  int *d_count;
  checkCudaErrors(cudaMalloc(&d_count, sizeof(int)));
  checkCudaErrors(cudaMemset(d_count, 0, sizeof(int)));

  // Try uncommenting one kernel call at a time
  xyzw_frequency<<<8, 256>>>(d_count, d_text, len);
  xyzw_frequency_thrust_device<<<1, 1>>>(d_count, d_text, len);
  checkCudaErrors(
      cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

  // xyzw_frequency_thrust_host(&count, h_text, len);
  
  std::chrono::steady_clock::time_point end=std::chrono::steady_clock::now();

  std::cout << "counted " << count
            << " instances of 'x', 'y', 'z', 'w' in \"" << filename << "\""
            << std::endl;
	

  std::cout << "v1: " << v1 << std::endl;
  std::cout << "v2: " << v2 << std::endl;
  Vector v3 = v1 + v2;
  std::cout << "v3 = v1 + v2: " << v3 << std::endl;
			
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

  checkCudaErrors(cudaFree(d_count));
  checkCudaErrors(cudaFree(d_text));
  
  

  return EXIT_SUCCESS;
}
