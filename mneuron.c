#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
//store and split training and testing data
double input[100][10];
double trainingSet[90][9];
double trainingOutput[90][1];
double testingSet[10][9];
double testingOutput[10][1];
 
double yEstimated[90][1];
double UntrainedMAE[90][1] ,UntrainedMMSE[90][1];
double sumofoutputlayer,sumofoutputerror, sumofsigmaerror;
double sumoutputinput1,sumoutputinput2,sumoutputinput3,sumoutputinput4;
double sumoutputinput1error,sumoutputinput2error,sumoutputinput3error,sumoutputinput4error;
double outputneuron[2][1], outputinput1[9][1],outputinput2[9][1],outputinput3[9][1],outputinput4[9][1];
//store weight and bias for input and output neuron
double outputweight[4],outputbias;
double weight1[9],bias1;
double weight2[9],bias2;
double weight3[9],bias3;
double weight4[9],bias4;
 
int iteration;
//Store untrain and trained MMSE for training set
double untrainMMSEforTraining, trainedMMSEforTraining;
 
//create array size to store all mae values inside
double plotarray[7000][1];
 
//function prototype
void readFile(char file[30]);
void inputNeuronRandomWeight(),outputNeuronRandomWeight();
void inputNeuron1(),inputNeuron2(),inputNeuron3(),inputNeuron4();
void trainOutputNeuron(),trainInputNeuron();
void outputNeuron();
void minMinSquareError();
void sigmoid();
void MAE();
void storeData();
 
int main (){
  char fileName[] = "fertility_Diagnosis_Data_Group5_8.txt";
    readFile(fileName);
    //get random weight for input and output neuron
    outputNeuronRandomWeight();
    inputNeuronRandomWeight();
    //calculate the linear regression for input layers
    inputNeuron1();
    inputNeuron2();
    inputNeuron3();
    inputNeuron4();
    //linear regression for output layer
    outputNeuron();
    //get yestimated
    sigmoid();
    //get untrained mmse for training set
    minMinSquareError();
    //check and train mae if needed
    MAE();
 
    storeData(); 
    //Calculate execution time
    double time_spent=0.0;
    //no of ticks from program setup from cpu
    clock_t begin = clock();
    //divide by CLOCKS_PER_SEC #1000000 to convert to seconds
    time_spent += (double)(begin) / CLOCKS_PER_SEC;
    printf("\nTotal Execution Time is %lf seconds", time_spent);
}
 
void storeData(){
    FILE* wptr;
    wptr=fopen("mdata.dat","w");
    if ((wptr=fopen("mdata.dat","w"))==NULL)
    {
        printf("file not found");
        exit(1);
    }
    else 
    {
        for (int i = 0; i < iteration; i++)
        { 
            fprintf(wptr,"%d %f\n",i+1,plotarray[i][0]);
        }
    }
    fclose(wptr);
}
 
void MAE(){
    double MAE,MMSE;
    for (int i = 0; i < 90; i++)
    {
        MAE += fabs((yEstimated[i][0] - trainingOutput[i][0])) / 90;
        MMSE += pow(yEstimated[i][0] - trainingOutput[i][0], 2) / 90;
    }
    //store mae value
    plotarray[iteration][0]=MAE;
    iteration++;
    if (MAE >=0.25)
    {
         printf("Iteration %d, MAE value: %f \n", iteration, MAE);  
         trainOutputNeuron();   
    }
    else
    {
       printf("Iteration %d, MAE value: %f Done! \n ", iteration, MAE);
       trainedMMSEforTraining=MMSE;
    }
}
 
void trainOutputNeuron()
{
    //sum of outputerror
    for (int i = 0; i < 90; i++)
    {
        sumofoutputerror = (yEstimated[i][0] - trainingOutput[i][0])/90;             
    }
    //sum of sigma error
    sumofsigmaerror = sumofoutputerror * (exp(sumofoutputlayer)/pow(1 + exp(sumofoutputlayer), 2));     
    //update output weights and bias
    outputweight[0] = sumofsigmaerror*sumoutputinput1;
    outputweight[1] = sumofsigmaerror*sumoutputinput2;
    outputweight[2] = sumofsigmaerror*sumoutputinput3;
    outputweight[3] = sumofsigmaerror*sumoutputinput4;    
    outputbias = 1*sumofsigmaerror;
    //back proporgate to train input neuron
    return trainInputNeuron();
}
 
void trainInputNeuron(){
    //get sum of input error for each neurons in input layer
    for(int i=0;i<9;i++)
    {    
        sumoutputinput1error = sumofsigmaerror* outputweight[0]*(exp(outputinput1[i][0]) / pow(1 + exp(outputinput1[i][0]), 2));
        sumoutputinput2error = sumofsigmaerror* outputweight[1]*(exp(outputinput2[i][0]) / pow(1 + exp(outputinput2[i][0]), 2)); 
        sumoutputinput3error = sumofsigmaerror* outputweight[2]*(exp(outputinput3[i][0]) / pow(1 + exp(outputinput3[i][0]), 2));
        sumoutputinput4error = sumofsigmaerror* outputweight[3]*(exp(outputinput4[i][0]) / pow(1 + exp(outputinput4[i][0]), 2));    
    }
    //update the weight and bias of input layer neurons
    for(int i=0;i<90;i++)
    {    
        for(int j=0;j<9;j++)
        {
            weight1[j] = sumoutputinput1error*trainingSet[i][j];
            weight2[j] = sumoutputinput2error*trainingSet[i][j];
            weight3[j] = sumoutputinput3error*trainingSet[i][j];
            weight4[j] = sumoutputinput4error*trainingSet[i][j];
        }
 
        bias1 = sumoutputinput1error*1;
        bias2 = sumoutputinput2error*1;
        bias3 = sumoutputinput3error*1;
        bias4 = sumoutputinput4error*1;
 
        // call the linear regression method again 
        inputNeuron1();
        inputNeuron2();
        inputNeuron3();
        inputNeuron4();
    }
    // get new MAE
    outputNeuron();
    sigmoid();
 
   return MAE();
}
 
void minMinSquareError()
{
    double sum = 0;
    for (int i = 0; i < 90; i++)
    {
         sum += pow(yEstimated[i][0] - trainingOutput[i][0], 2) / 90;
        
    }
    untrainMMSEforTraining=sum;
}
 
void sigmoid()
{
    for (int i = 0; i < 90; i++)
    {
        yEstimated[i][0] = 1 / (1 + exp(-sumofoutputlayer));
    }
 
}
 
void outputNeuron()
{
    for (int i = 0; i < 4; i++)
    {
       sumofoutputlayer = outputinput1[i][0]*outputweight[0] + outputinput2[i][0]*outputweight[1]+
                        outputinput3[i][0]*outputweight[2] + outputinput4[i][0]*outputweight[3]+outputbias;
    }
}
 
//hidden layer linear regression
void inputNeuron1()
{
    for (int i = 0; i < 9; i++)
    {
        outputinput1[i][0] = trainingSet[i][0] * weight1[0] + trainingSet[i][1] * weight1[1] + trainingSet[i][2] * weight1[2] 
        + trainingSet[i][3] * weight1[3] + trainingSet[i][4] * weight1[4] + trainingSet[i][5] * weight1[5] 
        + trainingSet[i][6] * weight1[6] + trainingSet[i][7] * weight1[7] + trainingSet[i][8] * weight1[8] + bias1;
        sumoutputinput1+=outputinput1[i][0];
    }
}
 
void inputNeuron2()
{
    for (int i = 0; i < 9; i++)
    {
        outputinput2[i][0] = trainingSet[i][0] * weight2[0] + trainingSet[i][1] * weight2[1] + trainingSet[i][2] * weight2[2] 
        + trainingSet[i][3] * weight2[3] + trainingSet[i][4] * weight2[4] + trainingSet[i][5] * weight2[5] 
        + trainingSet[i][6] * weight2[6] + trainingSet[i][7] * weight2[7] + trainingSet[i][8] * weight2[8] + bias2; 
        sumoutputinput2+=outputinput2[i][0];
    }
}
 
void inputNeuron3()
{
    for (int i = 0; i < 9; i++)
    {
        outputinput3[i][0] = trainingSet[i][0] * weight3[0] + trainingSet[i][1] * weight3[1] + trainingSet[i][2] * weight3[2] 
        + trainingSet[i][3] * weight3[3] + trainingSet[i][4] * weight3[4] + trainingSet[i][5] * weight3[5] 
        + trainingSet[i][6] * weight3[6] + trainingSet[i][7] * weight3[7] + trainingSet[i][8] * weight3[8] + bias3;
        sumoutputinput3+=outputinput3[i][0];
    }
}
 
void inputNeuron4()
{
    for (int i = 0; i < 9; i++)
    {
        outputinput4[i][0] = trainingSet[i][0] * weight4[0] + trainingSet[i][1] * weight4[1] + trainingSet[i][2] * weight4[2] 
        + trainingSet[i][3] * weight4[3] + trainingSet[i][4] * weight4[4] + trainingSet[i][5] * weight4[5] 
        + trainingSet[i][6] * weight4[6] + trainingSet[i][7] * weight4[7] + trainingSet[i][8] * weight4[8] + bias4; 
        sumoutputinput4+=outputinput4[i][0];
    }
}
 
//print random for weight and bias from -1,0,1
void inputNeuronRandomWeight()
{
    double w1, w2,w3, w4;
    int lower = -1, upper = 1, count = 10;
    srand(time(0));
    for (int i = 0; i < count; i++)
    {
        w1 = (rand() % (upper - lower + 1)) + lower;
        w2 = (rand() % (upper - lower + 1)) + lower;
        w3 = (rand() % (upper - lower + 1)) + lower;
        w4 = (rand() % (upper - lower + 1)) + lower;
 
        weight1[i] = w1;
        weight2[i] = w2;
        weight3[i] = w3;
        weight4[i] = w4;
        
        if (i == 9)
        {
            bias1 = w1;
            bias2 = w2;
            bias3 = w3;
            bias4 = w4;
        }
    }
}
 
void outputNeuronRandomWeight()
{
    double num, num2;
    int lower = -1, upper = 1, count = 4;
    srand(time(0));
    for (int i = 0; i < count; i++)
    {
        num = (rand() % (upper - lower + 1)) + lower;
        num2 = (rand() % (upper - lower + 1)) + lower;
 
        outputweight[i] = num;
        
        if (i == 9)
        {
            outputbias = num2;
        }
    }
}
//read fertility file
void readFile(char file[30])
{
    FILE *fptr;
    int count = 0;
    int trainingRow = 0, testingRow = 0;
    int row = 0, column = 0;
    char singleLine[100];
    if ((fptr = fopen(file, "r")) == NULL)
    {
        printf("file not found");
        exit(1);
    }
    else
    {
        while (!feof(fptr) && count < 100)
        {
            fgets(singleLine, 100, fptr);
            char *token = strtok(singleLine, ",");
            while (token != NULL)
            {
                double ret;
                ret = atof(token);
                input[row][column] = ret;
                token = strtok(NULL, ",");
                if (column == 9)
                {
                    row++;
                    column = -1;
                }
                column++;
            }
            count++;
        }
        //contains input from x0 to x8 - 9 inputs
        //Training Input
        for (int i = 0; i < 90; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                trainingSet[i][j] = input[i][j];
            }
        }
        //Training Result
        for (int i = 0; i < 90; i++)
        {
            for (int j = 0; j < 1; j++)
            {
                //input will take [i][9] which is the last output
                trainingOutput[i][j] = input[i][9];
            }
        }
        //Testing Input
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                testingSet[i][j] = input[i][j];
            }
        }
        //Testing Result
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 1; j++)
            {
                //input will take [i][9] which is the last output
                testingOutput[i][j] = input[i][9];               
            }
        }
    }
    fclose(fptr);
}
