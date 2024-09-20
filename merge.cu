#include <iostream>
#include <math.h>
#include <unistd.h>
#include <string>
#include <fcntl.h>
#include <random>

using namespace std;

unsigned long long write_disk(int in, void* tmp_buffer , unsigned long long size , unsigned long long file_offset)
{
    unsigned long long buffer_bias =0;
    unsigned long long left_size = size;
    char* buffer = (char*)tmp_buffer;
    while(left_size > 0) {
        //std::cout<< "***************************"<<std::endl;
        ssize_t return_value;
        return_value = pwrite(in, buffer+buffer_bias, left_size, file_offset);
        //std::cout << " pwrite return value : "<<return_value<<std::endl;
        if(return_value < 0)
        {
            int err = errno;
            fprintf(stderr, "pwrite failed: %s\n", strerror(err));
            
            std::cout<< " has write : " << size- left_size<<std::endl;
            std::cout << " left size : "<<left_size<<std::endl;
            std::cout << " file offset : "<<file_offset<<std::endl;
            return return_value;
        }
        buffer_bias += return_value;
        left_size -= return_value;
        file_offset += return_value;
        // std::cout << " left size : "<<left_size<<std::endl;
        // std::cout << " file offset : "<<file_offset<<std::endl;
    }
    return size;
}

int check_result(int fd,unsigned long long entry_num )
{
    std::cout << "check result" << std::endl;
    
    // for(int i = 0; i < 10; i++) {
    //     pread(fd, &value1, sizeof(int), i * sizeof(int));
    //     std::cout << "fd [ "<< i << "] = "<<value1 <<std::endl;
    // }
    unsigned long long tmp;
    pread(fd, &tmp, sizeof(int), (1742863087UL)*sizeof(int));
    std::mt19937 generator;
    for(unsigned long long i = 1; i < 1000; i++) {
        unsigned long long value1;
        unsigned long long value2;
        unsigned long long index1 = generator()%(entry_num -1 ); 
        unsigned long long index2 = generator()%(index1 ); 
        pread(fd, &value1, sizeof(int), (index1)*sizeof(int));
        pread(fd, &value2, sizeof(int), (index2)*sizeof(int));
        if(value2 > value1) {
            cout << "error" << endl;
            std::cout << "value [ "<<index1 <<"] = " << value1 <<std::endl;
            std::cout << "value [ "<<index2   <<"] = " << value2 <<std::endl;
            return -1;
        }
        value1 = value2;
    }
    std::cout << " =========== check pass ============ "<< std::endl;
    return 0;
}

void block_sort(int in,  unsigned long long file_offset , unsigned long long block_size, int thread_num) {
    // read data from input file
    
    int* tmp_buffer = (int * )malloc(block_size * sizeof(int));
    //std::cout<< "buffer size : "<<block_size * sizeof(int) << std::endl;
    //std::cout << " block_size  :"<< block_size  <<std::endl;
    //std::cout << " file_offset :"<< file_offset << std::endl;
    pread(in, tmp_buffer, block_size * sizeof(int), file_offset);
    //void *devPtr;
    //cudaMalloc(&devPtr, block_size * sizeof(int));
    //cudaMemcpy(devPtr, tmp_buffer, block_size * sizeof(int), cudaMemcpyHostToDevice);
    // sort data in memory
    ssize_t return_value;
    for(int i = 0; i < block_size; i++) {
        tmp_buffer[i] = i/1024;
    }
    return_value =  write_disk(in, tmp_buffer, block_size * sizeof(int), file_offset);
    
    //std::cout << " write size : " << block_size * sizeof(int) << endl;
    //std::cout << " pwrite return value : "<< return_value<<endl;
    // for(int i = 0; i < block_size; i++) {
    //     int  tmp;
    //     return_value = pread(in, &tmp, sizeof(int), i*sizeof(int));
    //     if(return_value != sizeof(int))
    //         std::cout << " pread return value : "<< return_value<<endl;
    //     if(tmp_buffer[i]!=tmp)
    //     {
    //         cout<< "start enequal!!! at "<<i<<endl;
    //         break;
    //     }
        
    // }
    
    // // write sorted data to output file
    // //cudaMemcpy(tmp_buffer, devPtr, block_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    

    // cout<< "value [ 1742863087]  "<<tmp<<endl;
    free(tmp_buffer);
    check_result(in,2UL*1024*1024*1024);
}

//* this function use binary search to find the partition index in the second file
void binary_find_partition_index(int in, int partitial_num, int* partitial_value, unsigned long long offset, unsigned long long entry_num , unsigned long long * partitial_index, unsigned long long* partitial_num2){
    int cursor_num = partitial_num*2  ;
    int* cursor_value = (int*)malloc(cursor_num * sizeof(int));
    unsigned long long* cursor_index = (unsigned long long*)malloc(cursor_num * sizeof(unsigned long long));
    int return_value;
    std::cout << " offset : "<< offset<<std::endl;
    for(unsigned long long i = 0; i < cursor_num ; i++) {
        cursor_index[i] = i*entry_num/cursor_num;
        return_value = pread(in, &(cursor_value[i]), sizeof(int), offset + cursor_index[i] * sizeof(int));
        // std::cout << " pread return value : "<<return_value<<std::endl;
        //std::cout<< " cursor["<<i<<"] index : "<<cursor_index[i]<<" value: "<<cursor_value[i]<<std::endl;
    }
    //! need to be checked
    unsigned long long left = 0;
    unsigned long long right = entry_num - 1;
    for(int i = 0; i < partitial_num ; i++) {  // (partitial_num -1 ) partition points
        //std::cout<< " **************************\n partiton num : "<<i<<std::endl;
        for(int j = 0; j < cursor_num; j++) {
            //std::cout<< " j= "<<j<<std::endl;
            if(partitial_value[i] <= cursor_value[j]) {
                right = cursor_index[j];
                if(i >0 && partitial_index[i-1] > left) {
                    left = partitial_index[i-1];
                }
                else if(j>0){
                    left = cursor_index[j - 1];
                }
                else{
                    left = 0;
                }
                break;
            }
        }
        //std::cout<<" left : "<< left<<" right : "<<right<<std::endl;
        while(left < right) {
            unsigned long long mid = (left + right) / 2;
            int mid_value ;
            pread(in, &mid_value, sizeof(int), offset + mid * sizeof(int));
            if(partitial_value[i] <= mid_value) {
                right = mid;
            }
            else {
                left = mid + 1;
            }
            //std::cout<<" left : "<< left<<" right : "<<right<<std::endl;
        }
        //getchar();
        partitial_index[i] =  left;
        if(i != 0) {
            partitial_num2[i-1] = partitial_index[i] - partitial_index[i-1];
        }

        if(i == partitial_num - 1) {
            partitial_num2[i] = entry_num - partitial_index[i];
        }
        
    }
}

__global__
void merge_kernel(int** a, int** b, int** c, unsigned long long* left_num_1, unsigned long long* left_num_2, unsigned long long* dest_num)
{
    //printf(" start kernel \n");
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int id = bid*32+tid;
    printf("Hello World from block %d and thread %d!\n", bid, tid);
    printf("left_num_1 : %llu , left_num_2 : %llu \n",left_num_1[id],left_num_2[id]);
    // int block_id = blockIdx.x; //todo need to be checked
    // //int start_offset = blockIdx.x * blockDim.x + threadIdx.x;  //todo need to be checked
    unsigned long long index_1 =0;
    unsigned long long index_2 =0;
    unsigned long long index_3 =0;
    while(1){
        if(left_num_1[id] == 0 && left_num_2[id] == 0 ){
                break;
            }else if(left_num_1[id] == 0){
                while(index_2 < left_num_2[id]){
                    c[id][index_3] = b[id][index_2];
                    index_2++;
                    index_3++;
                }
                break;
            }else if(left_num_2[id] == 0){
                while(index_1 < left_num_1[id]){
                    c[id][index_3] = a[id][index_1];
                    index_1++;
                    index_3++;
                }
                break;
        }
        if(a[id][index_1] <= b[id][index_2]) {
            c[id][index_3] = a[id][index_1];
            // printf("a[%d][%llu]  = %d \n ",id,index_1,a[id][index_1]);
            // printf("b[%d][%llu]  \n ",id,index_2);
            index_1++;
            index_3++;
        }
        else {
            c[id][index_3] = b[id][index_2];
            index_2++;
            index_3++;
        }
        //* judge when to escape : if a or b is empty
        if(index_1 == left_num_1[id]  || index_2 == left_num_2[id]  ) {
            //* move the rest of the data to the front
            

            if(index_1 == left_num_1[id]) {
                for(int i = 0; i < left_num_2[id] - index_2; i++) {
                    b[id][i] = b[id][i+ index_2];
                }
            }
            else {
                for(int i = 0; i < left_num_1[id] - index_1; i++) {
                    a[id][i ] = a[id][i+ index_1];
                }
            }
            break;
        }
    }
    printf(" index_1: %llu, index_2: %llu, index_3: %llu \n",index_1,index_2,index_3);
    // //std::cout<<"index1 : "<<index_1 << " index2: "<<index_2<<std::endl;
    left_num_1[id] -= index_1;
    left_num_2[id] -= index_2;
    dest_num[id] = index_3;
}

void parallel_merge(int in, int  out, unsigned long long fetch_num, unsigned long long thread_num, unsigned long long * partitial_index1,unsigned long long * partitial_index2,unsigned long long * partitial_num1 ,unsigned long long * partitial_num2 ,unsigned long long partitial_num,unsigned long long offset1,unsigned long long offset2,unsigned long long offset3 ) {
    int ** p_tmpbuffer1 = (int**)malloc(partitial_num * sizeof(int*));
    for(int i = 0; i < partitial_num; i++) {
        p_tmpbuffer1[i] = (int*)malloc(fetch_num * sizeof(int));
    }

    int ** p_tmpbuffer2 = (int**)malloc(partitial_num * sizeof(int*));
    for(int i = 0; i < partitial_num; i++) {
        p_tmpbuffer2[i] = (int*)malloc(fetch_num * sizeof(int));
    }
    int ** p_destbuffer = (int**)malloc(  partitial_num * sizeof(int*));
    for(int i = 0; i <  partitial_num; i++) {
        p_destbuffer[i] = (int*)malloc(2*fetch_num * sizeof(int));
    }
    unsigned long long * left_num_1 = (unsigned long long*)malloc(partitial_num * sizeof(unsigned long long));
    for(int i = 0; i < partitial_num; i++) {
        left_num_1[i] = partitial_num1[i];
    }
    unsigned long long * left_num_2 = (unsigned long long*)malloc(partitial_num * sizeof(unsigned long long));
    for(int i = 0; i < partitial_num; i++) {
        left_num_2[i] = partitial_num2[i];
    }
    unsigned long long * tmpbuffer1_left = (unsigned long long*)malloc(partitial_num * sizeof(unsigned long long));
    for(int i = 0; i < partitial_num; i++) {
        tmpbuffer1_left[i] = 0;
    }
    unsigned long long * tmpbuffer2_left = (unsigned long long*)malloc(partitial_num * sizeof(unsigned long long));
    for(int i = 0; i < partitial_num; i++) {
        tmpbuffer2_left[i] = 0;
    }
    unsigned long long * destbuffer_num = (unsigned long long*)malloc(partitial_num * sizeof(unsigned long long));
    for(int i = 0; ; i++) { // merge two block
        std::cout << "********************" << std::endl;
        std::cout<< " parrallel i: "<< i <<std::endl;
        bool flag = false;
        for(int j = 0; j < partitial_num; j++) {
            std::cout<<" left_num_1["<<j<<"] :"<<left_num_1[j]<<" left_num_2["<<j<<"] :"<<left_num_2[j]<<" tmpbuffer1_left["<<j<<"] :"<<tmpbuffer1_left[j]<<"  tmpbuffer2_left["<<j<<"] :"<< tmpbuffer2_left[j]<<std::endl;
            if(left_num_1[j] > 0 || left_num_2[j] > 0 || tmpbuffer1_left[j] > 0 || tmpbuffer2_left[j] > 0) {
                flag = true;
            }
        }
        if(!flag) {
            break;
        }
        //* phase 1: read data from input file to each tmp buffer
        std:: cout <<  " pahse 1 "<< std::endl;
        for(int j = 0; j < partitial_num; j++) {
            if(left_num_1[j] > 0 && tmpbuffer1_left[j] == 0) {
                int read_num = (left_num_1[j] > fetch_num) ? fetch_num : left_num_1[j];
                pread(in, p_tmpbuffer1[j], read_num * sizeof(int), offset1 + partitial_index1[j] * sizeof(int));
                left_num_1[j] -= read_num;
                tmpbuffer1_left[j] = read_num;
            }
            if(left_num_2[j] > 0 && tmpbuffer2_left[j] == 0) {
                int read_num = (left_num_2[j] > fetch_num) ? fetch_num : left_num_2[j];
                pread(in, p_tmpbuffer2[j], read_num * sizeof(int), offset2 + partitial_index2[j] * sizeof(int));
                left_num_2[j] -= read_num;
                tmpbuffer2_left[j] = read_num;
            }
        }
        //* phase 2: use gpu kernels parallel merge data in memory and move the unsorted data from back to front
        std:: cout<< " phase 2 "<<std::endl;
        int** devPtr1 ,** tmp_devPtr1;
        int** devPtr2 ,** tmp_devPtr2;
        int** devPtr3 ,** tmp_devPtr3;
        tmp_devPtr1 = (int**)malloc(sizeof(int*)*partitial_num);
        tmp_devPtr2 = (int**)malloc(sizeof(int*)*partitial_num);
        tmp_devPtr3 = (int**)malloc(sizeof(int*)*partitial_num);
        cudaMalloc(&devPtr1, sizeof(int*) * partitial_num);
        cudaMalloc(&devPtr2, sizeof(int*) * partitial_num);
        cudaMalloc(&devPtr3, sizeof(int*) * partitial_num);
        //cudaError_t err1 ;
        
        for(int j = 0; j < partitial_num; j++) {
            //std::cout << j << std::endl;
            cudaMalloc(&tmp_devPtr1[j], fetch_num * sizeof(int));
            cudaMalloc(&tmp_devPtr2[j], fetch_num * sizeof(int));
            cudaMalloc(&tmp_devPtr3[j], 2*fetch_num * sizeof(int));
            //std::cout << " cumemcpy 1"<< std::endl;
            cudaMemcpy(tmp_devPtr1[j], p_tmpbuffer1[j], fetch_num * sizeof(int), cudaMemcpyHostToDevice);
            
            // err1 = cudaGetLastError();
            // std::cout << "cuda error : "<< err1<<std::endl;
            //std::cout << " cumemcpy 2"<< std::endl;
            cudaMemcpy(tmp_devPtr2[j], p_tmpbuffer2[j], fetch_num * sizeof(int), cudaMemcpyHostToDevice);
            // err1 = cudaGetLastError();
            // std::cout << "cuda error : "<< err1<<std::endl;
        }
        cudaMemcpy(devPtr1, tmp_devPtr1, sizeof(int*) * partitial_num, cudaMemcpyHostToDevice);
        cudaMemcpy(devPtr2, tmp_devPtr2, sizeof(int*) * partitial_num, cudaMemcpyHostToDevice);
        cudaMemcpy(devPtr3, tmp_devPtr3, sizeof(int*) * partitial_num, cudaMemcpyHostToDevice);
        std::cout << " point "<< std::endl;
        unsigned long long* devPtr4;
        unsigned long long* devPtr5;
        unsigned long long* devPtr6;
        cudaMalloc(&devPtr4, partitial_num * sizeof(unsigned long long));
        cudaMalloc(&devPtr5, partitial_num * sizeof(unsigned long long));
        cudaMalloc(&devPtr6, partitial_num * sizeof(unsigned long long));
        cudaMemcpy(devPtr4, tmpbuffer1_left, partitial_num * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(devPtr5, tmpbuffer2_left, partitial_num * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(devPtr6, destbuffer_num, partitial_num * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        std::cout<< "partitial_num :  "<< partitial_num<< " fetch_num : "<< fetch_num << std::endl;
        std::cout << " kernel start " << std::endl;
        merge_kernel<<<partitial_num/32, 32>>>(devPtr1, devPtr2, devPtr3, devPtr4, devPtr5, devPtr6);
        cudaError_t err = cudaGetLastError();
        cudaDeviceSynchronize();
        std::cout << "cuda error : "<< err<<std::endl;
        std:: cout << " kernel complete "<< std::endl;
        
        for(int j = 0; j < partitial_num; j++) {
            //cudaMemcpy(p_tmpbuffer1[j], devPtr1[j], fetch_num * sizeof(int), cudaMemcpyDeviceToHost);
            //cudaMemcpy(p_tmpbuffer2[j], devPtr2[j], fetch_num * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(p_destbuffer[j], tmp_devPtr3[j], 2*fetch_num * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(tmp_devPtr1[j]);
            cudaFree(tmp_devPtr2[j]);
            cudaFree(tmp_devPtr3[j]);
        }
        cudaMemcpy(tmpbuffer1_left, devPtr4, partitial_num * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(tmpbuffer2_left, devPtr5, partitial_num * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(destbuffer_num, devPtr6, partitial_num * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaFree(devPtr4);
        cudaFree(devPtr5);
        cudaFree(devPtr6);
        cudaFree(devPtr1);
        cudaFree(devPtr2);
        cudaFree(devPtr3);
        free(tmp_devPtr1);
        free(tmp_devPtr2);
        free(tmp_devPtr3);
        for(int j = 0; j < partitial_num; j++) {
            std::cout<<" left_num_1["<<j<<"] :"<<left_num_1[j]<<" left_num_2["<<j<<"] :"<<left_num_2[j]<<" tmpbuffer1_left["<<j<<"] :"<<tmpbuffer1_left[j]<<"  tmpbuffer2_left["<<j<<"] :"<< tmpbuffer2_left[j]<<std::endl;
        }
        
        std::cout << " phase 3 " <<std::endl;
        //* phase 3: write data to output file 
        for(int j = 0; j < partitial_num; j++) {
            if(destbuffer_num[j] > 0) {
                write_disk(out, p_destbuffer[j], destbuffer_num[j] * sizeof(int), offset3 + partitial_index1[j] * sizeof(int));
                partitial_index1[j] += destbuffer_num[j];
            }
        }

    }
    for(int i = 0; i < partitial_num; i++) {
        free(p_tmpbuffer1[i]);
        free(p_tmpbuffer2[i]);
        free(p_destbuffer[i]);
    }
    free(p_tmpbuffer1);
    free(p_tmpbuffer2);
    free(p_destbuffer);
    free(left_num_1);
    free(left_num_2);
    free(tmpbuffer1_left);
    free(tmpbuffer2_left);
    free(destbuffer_num);
}

void merge_two(int in, int out, int block_size, int thread_num, unsigned long long offset1, unsigned long long offset2,unsigned long long offset3, unsigned long long entry_num1, unsigned long long entry_num2) {
    std::cout << " offset1: "<<offset1<<std::endl;
    std::cout << " offset2: "<<offset2<<std::endl;
    std::cout << " entry_num1: "<< entry_num1<< std::endl;
    std::cout << " entry_num2: "<< entry_num2<< std::endl;
    // find partitial value in a
    int partitial_num = thread_num ;
    int* partitial_value = (int*)malloc(partitial_num * sizeof(int));
    unsigned long long* partitial_index1 = (unsigned long long*)malloc(partitial_num * sizeof(unsigned long long));
    unsigned long long* partitial_index2 = (unsigned long long*)malloc(partitial_num * sizeof(unsigned long long));
    unsigned long long* partitial_num1 = (unsigned long long*)malloc(partitial_num * sizeof(unsigned long long));
    unsigned long long* partitial_num2 = (unsigned long long*)malloc(partitial_num * sizeof(unsigned long long));
    for(unsigned long long i = 0; i < partitial_num; i++) {
        partitial_index1[i] = i * entry_num1 / partitial_num;
        partitial_value[i] = 0;
        pread(in, &(partitial_value[i]), sizeof(int), offset1 + partitial_index1[i] * sizeof(int));
        if(i!= 0)
            partitial_num1[i-1] =  partitial_index1[i] - partitial_index1[i-1];
        if(i == partitial_num - 1)
            partitial_num1[i] = entry_num1 - partitial_index1[i] ;
    }
    // find partitial index in b
    binary_find_partition_index(in, partitial_num, partitial_value, offset2, entry_num2, partitial_index2, partitial_num2);
    // std::cout<< "partitial_num : " << partitial_num <<std::endl;
    // for(int i=0;i<partitial_num;i++)
    // {
    //     std::cout << " partitial_index2["<<i<<"]: "<<partitial_index2[i]<<"  num: "<<partitial_num2[i]<<std::endl;
    // }
    // std::cout<<"=================================="<<std::endl;
    // for(int i=0;i<partitial_num;i++)
    // {
    //     std::cout << " partitial_index1["<<i<<"]: "<<partitial_index1[i]<<" partitial_index2["<<i<<"]: "<<partitial_index2[i]<<"  num: "<<(int)(partitial_index2[i]-partitial_index1[i])<<std::endl;
    // }
    unsigned long long fetch_num = 16UL*1024*1024;
    parallel_merge(in,out,fetch_num,thread_num,partitial_index1, partitial_index2, partitial_num1 , partitial_num2 , partitial_num,offset1,offset2,offset3 );
    

}

void merge_pass(int in, int out, int block_num ,int thread_num, unsigned long long * offset_info, unsigned long long * out_offset_info,unsigned long long * entrynum_info, int block_size) {
    // read data from input file
    int merge_block_num = block_num  / 2;    //* if not the exponent of 2, the last block will be merged with the last block
    
    for(int i = 0; i < merge_block_num; i++) {
        std::cout<<"merge_block_num : "<<i<< std::endl;
        merge_two(in, out, block_size, thread_num, offset_info[2*i], offset_info[2*i+1],out_offset_info[i], entrynum_info[2*i], entrynum_info[2*i+1]);
    }
}


int merge_main(string fpath1, string fpath2, unsigned long long entry_num, unsigned long long block_size, int thread_num) {

    // phase 1: sort each block size data in memory 

    int fd_1;
    fd_1 = open(fpath1.c_str(), O_RDWR, 0);
    int fd_2;
    fd_2 = open(fpath2.c_str(),O_RDWR, 0);
    int block_num = (entry_num + block_size -1 )/block_size;
    for(int i = 0; i < block_num; i++) {
        if(i == block_num - 1) {
            block_sort(fd_1, i * block_size * sizeof(int), entry_num - i * block_size, thread_num);
        }
        else {
            block_sort(fd_1, i * block_size * sizeof(int), block_size, thread_num);
        }
    }
    
    //phase 2: merge sorted data
    unsigned long long offset_info_even[block_num];
    unsigned long long offset_info_odd[block_num];
    unsigned long long entrynum_info_even[block_num];
    unsigned long long entrynum_info_odd[block_num];
    for(int i = 0; i < block_num -1; i++) {
        offset_info_even[i] = i * block_size * sizeof(int);
        entrynum_info_even[i] = block_size;
        std::cout<< "offset_info_even [" <<i<<"] : "<<offset_info_even[i]<< std::endl;
        std::cout<< "entrynum_info_even [" <<i<<"] : "<<entrynum_info_even[i]<< std::endl;
    }
    offset_info_even[block_num - 1] = (block_num - 1) * block_size * sizeof(int);
    entrynum_info_even[block_num - 1] = entry_num - (block_num - 1) * block_size;
    int block_num_even = block_num;
    int block_num_odd ;
    //int block_size; //todo need to be calculated
    int pass_num = floor(log2(block_num));
    std::cout<<" total merge pass : "<<pass_num<<std::endl;
    for(int i=0; i<pass_num; i++) {
        std::cout<<"merge pass : "<<i<<std::endl;
        if(i % 2 == 0) {
            block_num_odd = (block_num_even +1) / 2;
            for(int j=0;j<block_num_odd;j++){
                offset_info_odd[j] = offset_info_even[2*j];
                if(j == block_num_odd-1 && block_num_even < block_num_odd*2){
                    entrynum_info_odd[j] = entrynum_info_even[2*j] ;
                }
                else{
                    entrynum_info_odd[j] = entrynum_info_even[2*j+1] + entrynum_info_even[2*j];
                }
                
            }
            merge_pass(fd_1, fd_2, block_num_even, thread_num, offset_info_even,offset_info_odd, entrynum_info_even,block_size);
        }
        else {
            block_num_even = (block_num_odd +1) / 2;
            for(int j=0;j<block_num_odd;j++){
                offset_info_even[j] = offset_info_odd[2*j];
                if(j == block_num_even-1 && block_num_odd < block_num_even*2){
                    entrynum_info_even[j] = entrynum_info_odd[2*j] ;
                }
                else{
                    entrynum_info_even[j] = entrynum_info_odd[2*j] + entrynum_info_odd[2*j+1];
                }
                
            }
            merge_pass(fd_2, fd_1, block_num_odd, thread_num, offset_info_odd,offset_info_even, entrynum_info_odd,block_size);
        }
    }
    close(fd_1);
    close(fd_2);
    return 0;
}



int main(void)
{
    string s1="/home/szy/ssd1/testfile1";
    string s2="/home/szy/ssd1/testfile2";
    merge_main(s1,s2,4UL*1024*1024*1024,2UL*1024*1024*1024,128);
    int fd = open(s1.c_str(), O_RDWR, 0);
    check_result(fd,4UL*1024*1024*1024);
    return 0;
}