#include "LeNet-5/lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>

#include "src/rlwe_sife.h"
#include "src/sample.h"
#include "src/params.h"

#define FILE_TRAIN_IMAGE	"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL	"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 			"model.dat"
#define COUNT_TRAIN			60000
#if defined(ORI) || defined(PLAIN)
#define COUNT_TEST			10000
#endif
#if defined(CPU) || defined(GPU)
#define COUNT_TEST			20
#endif


int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const int sequence, const char data_file[], const char label_file[])
{
	int temp=0;
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fseek(fp_image, sizeof(*data)*sequence, SEEK_CUR);
	fseek(fp_label, sequence, SEEK_CUR);
	temp=fread(data, sizeof(*data)*count, 1, fp_image);
	temp=fread(label, count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(lenet, train_data + i, train_label + i, batch_size);
		if (i * 100 / total_size > percent)
			printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
	}
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict(lenet, test_data[i], 10);
		right += l == p;
		if (i * 100 / total_size > percent)
			printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}

#if defined(PLAIN) || defined(CPU) || defined(GPU)
int sec_testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
#ifdef TIME2
    uint64_t CLOCK1, CLOCK2;
    double setup_time=0, keygen_time=0, load_time=0, predict_time=0;
#endif   
	uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N];
	uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N];

	printf("start\t");
	printf("UNKNOWN: %d\t", UNKNOWN);
	printf("TERMS: %d\n", TERMS);
#ifdef CPU
#ifdef TIME2
    CLOCK1=cpucycles();
#endif   
    rlwe_sife_setup(mpk, msk);
#ifdef TIME2
    CLOCK2=cpucycles();
    setup_time = (double)CLOCK2 - CLOCK1;
#endif   
#endif
#ifdef GPU
#ifdef TIME2
    CLOCK1=cpucycles();
#endif   
    rlwe_sife_setup_gui(mpk, msk);
#ifdef TIME2
    CLOCK2=cpucycles();
    setup_time = (double)CLOCK2 - CLOCK1;
#endif   
#endif
	//printf("setup finished\n");



#ifdef CPU
#ifdef TIME2
    CLOCK1=cpucycles();
#endif   

#ifdef TIME2
    CLOCK2=cpucycles();
    keygen_time = (double)CLOCK2 - CLOCK1;
#endif   
#endif



	for (int i = 0; i < total_size; ++i)
	{
#ifdef TIME2
    CLOCK1=cpucycles();
#endif   
		//pre-processing part
		sec_Feature features = { 0 };

#ifdef PLAIN
		sec_load_input(&features, test_data[i]);
#endif
#if defined(CPU) || defined(GPU)
		sec_load_input(&features, test_data[i], mpk);
#endif
	//printf("load finished\n");

#ifdef TIME2
    CLOCK2=cpucycles();
    load_time = (double)CLOCK2 - CLOCK1;
    CLOCK1=cpucycles();
#endif   
		//inference part
#ifdef PLAIN
		uint8 l = test_label[i];
		int p = sec_Predict(lenet, &features, 10);
#endif
#if defined(CPU) || defined(GPU)
		uint8 l = test_label[i];
		int p = sec_Predict(lenet, &features, 10, msk);
#endif
	//printf("predict finished\n");

#ifdef TIME2
    CLOCK2=cpucycles();
    predict_time = (double)CLOCK2 - CLOCK1;
    //printf("setup time: \t %.6f ms\t", setup_time/CLOCKS_PER_MS);
    //printf("load time: \t %.6f ms\t", load_time/CLOCKS_PER_MS);
    //printf("predict time: \t %.6f ms\n", predict_time/CLOCKS_PER_MS);
    FILE *fp_performance2 = fopen("performance in detail.txt", "at+");
	fprintf(fp_performance2, "label:%d,predict:%d\t", l, p);
    fprintf(fp_performance2, "Setup time:\t%.6f\tms\t", setup_time/CLOCKS_PER_MS);
    fprintf(fp_performance2, "Load time:\t%.6f\tms\t", load_time/CLOCKS_PER_MS);
    fprintf(fp_performance2, "Predict time:\t%.6f\tms\n", predict_time/CLOCKS_PER_MS);
    fclose(fp_performance2);
#endif   
		right += l == p;
		//printf("label: %d , predict: %d \n", l, p);
		//if (i * 100 / total_size > percent)
		//	printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}
#endif   

int save(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

int load(LeNet5 *lenet, char filename[])
{
	int temp=0;
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	temp=fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}



void foo(int sequence)
{
#ifdef TIME 
    uint64_t CLOCK1, CLOCK2;
    double train_time=0, test_time=0;
    CLOCK1=cpucycles();
#endif   
	int right =0;
	//image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	//uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	// if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	// {
	// 	printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
	// 	free(train_data);
	// 	free(train_label);
	// }
	if (read_data(test_data, test_label, COUNT_TEST, sequence, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(test_data);
		free(test_label);
	}


	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	if (load(lenet, LENET_FILE))
		Initial(lenet);

// #ifdef TIME 
//     CLOCK1=cpucycles();
// #endif   
// 	//int batches[] = { 300 };
// 	//for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
// 	//	training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);
// #ifdef TIME 
//     CLOCK2=cpucycles();
//     train_time += (double)CLOCK2 - CLOCK1;
// #endif   

#ifdef TIME 
    CLOCK1=cpucycles();
#endif   
#ifdef ORI
	right = testing(lenet, test_data, test_label, COUNT_TEST);
#endif   
#if defined(PLAIN) || defined(CPU) || defined(GPU)
	right = sec_testing(lenet, test_data, test_label, COUNT_TEST);
#endif   
#ifdef TIME 
    CLOCK2=cpucycles();
    test_time += (double)CLOCK2 - CLOCK1;
    //printf("Train time: \t \t %.6f ms\n", train_time/CLOCKS_PER_MS);
    printf("Inference time: \t %.6f ms\n", test_time/CLOCKS_PER_MS);
    FILE *fp_performance = fopen("performance.txt", "at+");
    fprintf(fp_performance, "Inference time:\t%.6f\tms\n", test_time/CLOCKS_PER_MS);
    fclose(fp_performance);
#endif   
	printf("%d/%d\n", right, COUNT_TEST);

	//save(lenet, LENET_FILE);
	free(lenet);
	// free(train_data);
	// free(train_label);
	free(test_data);
	free(test_label);
	//system("pause");
	//system("read -p 'Press Enter to continue...' var");
}

int main(int argc, char* argv[])
{
	int sequence = 0;
	sequence = atoi(argv[1]);
	foo(sequence);
	return 0;
}