#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
 
//GLOBAL ARRAY 
//Split the input and output into different array
double input[100][10];
double trainingSet[90][9];
double trainingOutput[90][1];
double testingSet[10][9];
double testingOutput[10][1];
//Linear Regression for TRAINING AND TESTING SET
double zLinear[90][1]; 
double testzLinear[10][1]; 
//Store yEstimated for TRAINING AND TESTING SET
double yEstimated[90][1]; 
double testyEstimated[10][1]; 
//Store MAE values for TRAINING SET
double MAE[90][1];     
//Store UNTRAINED MMSE values for TRAINING AND TESTING SET
double MMSE[90][1];     
double testMMSE[10][1]; 
//Store the MAE iteration for both training and testing set MAX LIMIT 1999 row
double plotarray[2000][1];
//STORE TRAINED MMSE VALUE FOR TRAINING SET 
double trainedtestzLinear[10][1];
double trainedtestyEstimated[10][1]; 
double trainedtestMMSE[10][1]; 

//GLOBAL VARIABLES
//Training speed
double alpha = 0.05;
//Initialize WEIGHT AND BIAS TRAINING set 
double weight[9], bias = 0;
//Split up to count the diff iteration for both training and testing set 
int iteration;
//Store untrain and trained MMSE for both testing and training set
double untrainMMSEforTraining, trainedMMSEforTraining;
double untrainMMSEforTesting, trainedMMSEforTesting;
 
//FUNCTION PROTOTYPE
//Read fertility input file
void readFile(char file[30]);
//Randomize from -1,0,1
void printRandoms(); 
//Linear Regression for TRAINING, TESTING SET
void linearRegression();    
void testlinearRegression(); 
//Sigmoid for TRAINING, TESTING SET
void sigmoid();     
void testsigmoid(); 
//MAE for TRAINING SET
void meanAbsoluteError();     
void trainmeanAbsoluteError();
//Untrained MMSE for TRAINING, TESTING SET
void minMinSquareError();            
void testminMinSquareError();     
//TRAINED MMSE FOR TESTING SET
void trainedtestlinearRegression(); 
void trainedtestsigmoid(); 
void trainedtestminMinSquareError();   
//Store MAE data for TRAINING
void storeData();
//Confusion Matrix for TRAINING, TESTING SET
void confusionmatrix();
void testconfusionmatrix();
//CALCULATE MMSE FOR UNTRAINED AND TRAINED TESTING SET
void MMSEUntrainedTestingSet();
void MMSETrainedTestingSet();
 
int main()
{
    char fileName[] = "fertility_Diagnosis_Data_Group5_8.txt";
    readFile(fileName);
    printRandoms();
    
    //TRAINING SET CALCULATE THE UNNTRAINED MMSE 
    linearRegression();
    sigmoid();
    minMinSquareError();
    //CALCULATE UNTRAINED MMSE FROM TESTING SET 
    MMSEUntrainedTestingSet();
    //UPDATE NEW WEIGHT AND BIAS FROM TRAINING SET 
    meanAbsoluteError();
    //GET NEW MMSE FROM TESTING SET 
    MMSETrainedTestingSet();

    printf("\n---TRAINING SET---\n");
    printf("Untrained MMSE: %f",untrainMMSEforTraining);
    printf("\nTrained MMSE: %f\n",trainedMMSEforTraining);
    confusionmatrix();
 
    printf("\n---TESTING SET---\n");
    printf("Untrained MMSE: %f",untrainMMSEforTesting);
    printf("\nTrained MMSE: %f\n",trainedMMSEforTesting);
    testconfusionmatrix();
    
    storeData(); 
 
    //Calculate execution time
    double time_spent=0.0;
    //no of ticks from program setup from cpu
    clock_t begin = clock();
    //divide by CLOCKS_PER_SEC #1000000 to convert to seconds
    time_spent += (double)(begin) / CLOCKS_PER_SEC;
    printf("\nTotal Execution Time is %lf seconds", time_spent);
}
//Store MAE value for each iteration for TRAINING SET
void storeData(){
    FILE* wptr;
    wptr=fopen("data.dat","w");
    if ((wptr=fopen("data.dat","w"))==NULL)
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
//MAE - MEAN ABSOLUTE ERROR >0.25 UPDATE WEIGHT
//MAE for TRAINING SET
void meanAbsoluteError()
{
    double MAEsum = 0.0, MMSEsum = 0.0;
    //1 iteration = 90 loops
    for (int i = 0; i < 90; i++)
    {
        MAE[i][0] = fabs((yEstimated[i][0] - trainingOutput[i][0])) / 90;
        MAEsum += MAE[i][0];
          
        MMSE[i][0] = pow(yEstimated[i][0] - trainingOutput[i][0], 2) / 90;
        MMSEsum += MMSE[i][0];
    }
    //store the sum in an array size is based on number of iteration limit is 2000
    plotarray[iteration][0]=MAEsum;
    iteration++;
    if (MAEsum >= 0.25)
    {
        printf("Iteration %d, MAE value: %f \n", iteration, MAEsum);
        trainmeanAbsoluteError();
    }
    else
    {
        printf("Iteration %d, MAE value: %f Done! \n ", iteration, MAEsum);
        //SHOW TRAINED MMSE FOR TRAINING SET 
        trainedMMSEforTraining=MMSEsum;
    }
}
//BACK PROPORGATION for TRAINING set
void trainmeanAbsoluteError()
{
    for (int i = 0; i < 90; i++)
    {
        for (int x = 0; x < 9; x++)
        {
            weight[x] = weight[x] - (alpha * (((yEstimated[i][0] - trainingOutput[i][0]) / 90) * 
            (exp(zLinear[i][0]) / pow(1 + exp(zLinear[i][0]), 2)) 
            * trainingSet[i][x]));   
        }                                       
        bias = bias - (alpha * ((yEstimated[i][0] - trainingOutput[i][0]) / 90) * 
        (exp(zLinear[i][0]) / pow(1 + exp(zLinear[i][0]), 2)) * 1);
        
        linearRegression();
        sigmoid();
    }
    return meanAbsoluteError();
}
//Feedforward
//Linear Regression for TRAINING SET
void linearRegression()
{
    //printf("\nLinear regression\n");
    for (int i = 0; i < 90; i++)
    {
        zLinear[i][0] = trainingSet[i][0] * weight[0] + trainingSet[i][1] * weight[1] + trainingSet[i][2] * weight[2] 
        + trainingSet[i][3] * weight[3] + trainingSet[i][4] * weight[4] + trainingSet[i][5] * weight[5] 
        + trainingSet[i][6] * weight[6] + trainingSet[i][7] * weight[7] + trainingSet[i][8] * weight[8] + bias;
    }
}
//Sigmoid for untrained TRAINING SET
void sigmoid()
{
    for (int i = 0; i < 90; i++)
    {
        yEstimated[i][0] = 1 / (1 + exp(-zLinear[i][0]));
    }
}
//UNTRAINED MMSE minimum min square error for TRAINING SET
void minMinSquareError()
{
    double sum = 0.0;
    for (int i = 0; i < 90; i++)
    {
        MMSE[i][0] = pow(yEstimated[i][0] - trainingOutput[i][0], 2) / 90;
        sum += MMSE[i][0];
    }
    untrainMMSEforTraining=sum;
}

//TESTING SET 
//FEED FORWARD
void MMSEUntrainedTestingSet(){
    testlinearRegression();
    testsigmoid();
    testminMinSquareError();
}
//Linear Regression for  UNTRAINED TESTING SET
void testlinearRegression()
{
    for (int i = 0; i < 10; i++)
    {
        testzLinear[i][0] = testingSet[i][0] * weight[0] + testingSet[i][1] * weight[1] 
        + testingSet[i][2] * weight[2] + testingSet[i][3] * weight[3] + testingSet[i][4] * weight[4] + testingSet[i][5] * weight[5] 
        + testingSet[i][6] * weight[6] + testingSet[i][7] * weight[7] + testingSet[i][8] * weight[8] + bias;
    }

}
//Sigmoid for TESTING SET
void testsigmoid()
{
    for (int i = 0; i < 10; i++)
    {
        testyEstimated[i][0] = 1 / (1 + exp(-testzLinear[i][0]));
    }
   
}
//UNTRAINED MMSE minimum min square error for TESTING SET
void testminMinSquareError()
{
    double sum = 0.0;
    for (int i = 0; i < 10; i++)
    {
        testMMSE[i][0] = pow(testyEstimated[i][0] - testingOutput[i][0], 2) / 10;
        sum += testMMSE[i][0];
    }
    untrainMMSEforTesting=sum;
}

void MMSETrainedTestingSet(){
    trainedtestlinearRegression();
    trainedtestsigmoid();
    trainedtestminMinSquareError();
}

void trainedtestlinearRegression()
{
    for (int i = 0; i < 10; i++)
    {
        trainedtestzLinear[i][0] = testingSet[i][0] * weight[0] + testingSet[i][1] * weight[1] 
        + testingSet[i][2] * weight[2] + testingSet[i][3] * weight[3] + testingSet[i][4] * weight[4] + testingSet[i][5] * weight[5] 
        + testingSet[i][6] * weight[6] + testingSet[i][7] * weight[7] + testingSet[i][8] * weight[8] + bias;
    }
}
//Sigmoid for TESTING SET
void trainedtestsigmoid()
{
    for (int i = 0; i < 10; i++)
    {
        trainedtestyEstimated[i][0] = 1 / (1 + exp(-trainedtestzLinear[i][0]));
    }
}

//TRAINED MMSE minimum min square error for TESTING SET
void trainedtestminMinSquareError()
{
    double sum = 0;
    for (int i = 0; i < 10; i++)
    {
        trainedtestMMSE[i][0] = pow(trainedtestyEstimated[i][0] - testingOutput[i][0], 2) / 10;
        sum += trainedtestMMSE[i][0];
    }
    trainedMMSEforTesting=sum;
}
//print random for weight and bias from -1,0,1
void printRandoms()
{
    double num, num2;
    int lower = -1, upper = 1, count = 10;
    srand(time(0));
    for (int i = 0; i < count; i++)
    {
        num = (rand() % (upper - lower + 1)) + lower;
        num2 = (rand() % (upper - lower + 1)) + lower;
        weight[i] = num;
        if (i == 9)
        {
            bias = num2;
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
//Confusion Matrix for TRAINING SET
void confusionmatrix()
{
    int i = 0;
    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;
    for (i = 0; i < 90; i++)
    {
        if (trainingOutput[i][0] == 1)
        {
            if (yEstimated[i][0] >= 0.5)
            tp++;
            else
            fn++;
        }
        else
        {
            if (yEstimated[i][0] >= 0.5)
            fp++;
            else
            tn++;
        }
    }
    printf("Confusion Matrix:\nTP: %d TN:%d FP:%d FN:%d\n", tp, tn, fp, fn);
}
//Confusion Matrix for TESTING SET
void testconfusionmatrix()
{
    int i = 0;
    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;
    for (i = 0; i < 10; i++)
    {
        if (testingOutput[i][0] == 1)
        {
            if (testyEstimated[i][0] >= 0.5)
            tp++;
            else
            fn++;
        }
        else
        {
            if (testyEstimated[i][0] >= 0.5)
            fp++;
            else
            tn++;
        }
    }
    printf("Confusion Matrix: \nTP: %d TN:%d FP:%d FN:%d\n", tp, tn, fp, fn);
}

