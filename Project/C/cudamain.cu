#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <sys/sysinfo.h>
#include <pthread.h>

#include "workshop.h"

#define TUPLESIZE 10
#define VECTORINITIALSIZE 1000
#define MAXCLASS 101
#define CORES 24

#define TIMER_START() gettimeofday(&t1, NULL)
#define TIMER_STOP()         \
    gettimeofday(&t2, NULL); \
    timersub(&t2, &t1, &t);  \
    time_delta = (float)t.tv_sec + t.tv_usec / 1000000.0

struct timeval t1, t2, t;
float time_delta;

typedef struct
{
    int type;
    int stars;
    int tuple[TUPLESIZE];
} Rule;

typedef struct
{
    int size;
    int maxsize;
    Rule *rules;
} Rulevector;

typedef struct
{
    int *tuple;
} Transaction;

typedef struct
{
    int size;
    int maxsize;
    Transaction *transactions;
} Transactionvector;

typedef struct
{
    int type;
    Rulevector *vector;
} Classification;

typedef struct
{
    int size;
    int maxsize;
    Classification **classes;
} Classvector;

typedef struct
{
    Classvector *vector;
} Solver;

Rule create_rule(char *line)
{
    char aux[strlen(line)];
    strcpy(aux, line);
    Rule r;
    r.stars = 0;

    char *token;
    token = strtok(aux, ";");
    int index = 0;
    while (token != NULL)
    {
        if (strcmp(token, "*") == 0)
        {
            r.tuple[index++] = -1;
            r.stars++;
        }
        else
        {
            if (index == TUPLESIZE)
                r.type = atoi(token);
            else
                r.tuple[index++] = atoi(token);
        }
        token = strtok(NULL, ";");
    }
    return r;
}

Transaction create_transaction(char *line)
{
    char aux[strlen(line)];
    strcpy(aux, line);
    Transaction t;
    t.tuple = (int *)calloc(TUPLESIZE, sizeof(int));

    char *token;
    token = strtok(aux, ";");
    int index = 0;
    while (token != NULL)
    {
        t.tuple[index++] = atoi(token);
        token = strtok(NULL, ";");
    }
    return t;
}

Rulevector *create_rule_vector()
{
    Rulevector *v = (Rulevector *)calloc(1, sizeof(Rulevector));
    v->maxsize = VECTORINITIALSIZE;
    v->rules = (Rule *)calloc(v->maxsize, sizeof(Rule));
    v->size = 0;
    return v;
}

Classification *create_classification(int type)
{
    Classification *c = (Classification *)malloc(sizeof(Classification));
    c->type = type;
    c->vector = create_rule_vector();
    return c;
}

Transactionvector *create_transaction_vector()
{
    Transactionvector *v = (Transactionvector *)malloc(sizeof(Transactionvector));
    v->maxsize = VECTORINITIALSIZE;
    v->transactions = (Transaction *)calloc(v->maxsize, sizeof(Transaction));
    v->size = 0;
    return v;
}

Classvector *create_class_vector()
{
    Classvector *v = (Classvector *)malloc(sizeof(Classvector));
    v->size = 0;
    v->maxsize = MAXCLASS;
    v->classes = (Classification **)calloc(v->maxsize, sizeof(Classification));
    return v;
}

Rulevector *addRuleToVector(Rulevector *v, Rule r)
{
    if (v->size == v->maxsize)
    {
        v->maxsize *= 2;
        v->rules = (Rule *)realloc(v->rules, v->maxsize * sizeof(Rule));
    }
    memcpy(&v->rules[v->size++], &r, sizeof(Rule));
    //v->rules[v->size++] = r;
    return v;
}

Transactionvector *addTransactionToVector(Transactionvector *v, Transaction t)
{
    if (v->size == v->maxsize)
    {
        v->maxsize *= 2;
        v->transactions = (Transaction *)realloc(v->transactions, v->maxsize * sizeof(Transaction));
    }
    v->transactions[v->size++] = t;
    return v;
}

Classvector *addClassToVector(Classvector *v, Classification *c)
{
    if (v->size == v->maxsize)
    {
        v->maxsize *= 2;
        v->classes = (Classification **)realloc(v->classes, v->maxsize * sizeof(Classification));
    }
    v->classes[v->size++] = c;
    return v;
}

void destroyRulevector(Rulevector *v)
{
    free(v->rules);
    free(v);
}

void destroyTransaction(Transaction t)
{
    free(t.tuple);
}

void destroyTransactionvector(Transactionvector *v)
{
    for (int i = 0; i < v->size; i++)
    {
        destroyTransaction(v->transactions[i]);
    }
    free(v->transactions);
    free(v);
}

void destroy_classification(Classification *c)
{
    destroyRulevector(c->vector);
    free(c);
}

void destroyClassvector(Classvector *v)
{
    for (int i = 0; i < v->size; i++)
    {
        destroy_classification(v->classes[i]);
    }
    free(v->classes);
    free(v);
}

int NumberLen(int n)
{
    if (n == 0)
        return 1;
    else
        return floor(log10(abs(n))) + 1;
}

int rule_matches(Rule r, Transaction t)
{
    for (int i = 0; i < TUPLESIZE; i++)
    {
        if (r.tuple[i] == -1)
        {
            continue;
        }
        if (r.tuple[i] != t.tuple[i])
        {
            return 0;
        }
    }
    return 1;
}

int classification_matches(Classification *c, Transaction t)
{
    for (int i = 0; i < c->vector->size; i++)
    {
        if (rule_matches(c->vector->rules[i], t))
        {
            return 1;
        }
    }
    return 0;
}

int compareRuleTo(Rule r1, Rule r2)
{
    int n = r1.type - r2.type;
    return n == 0 ? r2.stars - r1.stars : n;
}

char *ruleToString(Rule r)
{
    char *res = (char *)calloc(1, 70);
    int offset = 0;
    for (int i = 0; i < TUPLESIZE; i++)
    {
        if (r.tuple[i] == -1)
        {
            sprintf(res + offset, "%s", "*;");
            offset += 2;
        }
        else
        {
            int nDigits = NumberLen(r.tuple[i]);
            sprintf(res + offset, "%d;", r.tuple[i]);
            offset += 1 + nDigits;
        }
    }
    sprintf(res + offset, "%d", r.type);
    return res;
}

int ruleEqual(Rule r1, Rule r2)
{
    if (compareRuleTo(r1, r2) == 0)
    {
        for (int i = 0; i < TUPLESIZE; i++)
        {
            if (r1.tuple[i] != r2.tuple[i])
            {
                return 0;
            }
        }
        return 1;
    }
    return 0;
}

Classification *addRule(Classification *c, Rule r)
{
    if (c->vector->size == 0 || !ruleEqual(c->vector->rules[c->vector->size - 1], r))
    {
        c->vector = addRuleToVector(c->vector, r);
    }
    return c;
}

char *transactionToString(Transaction t)
{
    char *res = (char *)calloc(1, 70);
    int offset = 0;
    for (int i = 0; i < TUPLESIZE; i++)
    {
        int nDigits;
        if (t.tuple[i] == 0)
            nDigits = 1;
        else
            nDigits = floor(log10(abs(t.tuple[i]))) + 1;
        sprintf(res + offset, "%d;", t.tuple[i]);
        offset += 1 + nDigits;
    }
    return res;
}

// A utility function to swap two elements
void swap(Rule *a, Rule *b)
{
    Rule t = *a;
    *a = *b;
    *b = t;
}

/* This function takes last element as pivot, places 
   the pivot element at its correct position in sorted 
    array, and places all smaller (smaller than pivot) 
   to left of pivot and all greater elements to right 
   of pivot */
int partition(Rule arr[], int low, int high)
{

    Rule pivot = arr[high]; // pivot

    int i = (low - 1); // Index of smaller element

    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (compareRuleTo(arr[j], pivot) < 0)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

/* The main function that implements QuickSort 
 arr[] --> Array to be sorted, 
  low  --> Starting index, 
  high  --> Ending index */
void quickSort(Rule arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now 
           at right place */
        int pi = partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void writeRulesToFile(Rulevector *v, char *filename)
{
    FILE *fp;
    fp = fopen(filename, "w");

    for (int i = 0; i < v->size; i++)
    {
        fputs(strcat(ruleToString(v->rules[i]), "\n"), fp);
    }
    fclose(fp);
}

Solver *create_solver(char *rulepath)
{
    TIMER_START();
    Solver *s = (Solver *)malloc(sizeof(Solver));
    s->vector = create_class_vector();

    FILE *fp;
    fp = fopen(rulepath, "r");

    Rulevector *rv = create_rule_vector();

    char *line = NULL;
    ssize_t read = 0;
    size_t len = 0;
    while ((read = getline(&line, &len, fp)) > 0)
    {
        rv = addRuleToVector(rv, create_rule(line));
    }
    free(line);
    fclose(fp);

    TIMER_STOP();
    printf("Rules Loaded. Took %f seconds\n",time_delta);

    TIMER_START();
    quickSort(rv->rules, 0, rv->size - 1);
    TIMER_STOP();
    printf("Rules Sorted. Took %f seconds\n",time_delta);

    TIMER_START();
    Classification *c = create_classification(rv->rules[0].type);
    c = addRule(c, rv->rules[0]);
    s->vector = addClassToVector(s->vector, c);

    for (int i = 1; i < rv->size; i++)
    {
        if (rv->rules[i].type != c->type)
        {
            // printf("Class %d size = %d\n", c->type, c->vector->size);
            c = create_classification(rv->rules[i].type);
            c = addRule(c, rv->rules[i]);
            s->vector = addClassToVector(s->vector, c);
        }
        else
        {
            c = addRule(c, rv->rules[i]);
        }
        //printf("Number of Rules of type %d = %d\n", c.type, c.vector->size);
    }
    destroyRulevector(rv);

    TIMER_STOP();
    printf("Classifications created. Took %f seconds\n",time_delta);

    return s;
}

void solve(Solver *s, char *in, char *out)
{
    TIMER_START();
    Transactionvector *tv = create_transaction_vector();

    FILE *infile;
    infile = fopen(in, "r");

    char *line = NULL;
    ssize_t read = 0;
    size_t len = 0;
    while ((read = getline(&line, &len, infile)) > 0)
    {
        tv = addTransactionToVector(tv, create_transaction(line));
    }
    free(line);
    fclose(infile);

    TIMER_STOP();
    printf("Time to read Transactions = %f seconds\n",time_delta);

    FILE *outfile;
    outfile = fopen(out, "w");

    char buffer[BUFSIZ] = {0};
    int offset = 0;

    TIMER_START();

    for (int i = 0; i < tv->size; i++)
    {
        char *temp = transactionToString(tv->transactions[i]);
        for (int j = 0; j < s->vector->size; j++)
        {
            if (classification_matches(s->vector->classes[j], tv->transactions[i]))
            {
                sprintf(buffer + offset, "%s%d\n", temp, s->vector->classes[j]->type);
                offset += strlen(temp) + NumberLen(s->vector->classes[j]->type) + 1;
            }
            if (offset >= BUFSIZ - 60)
            {
                fputs(buffer, outfile);
                memset(buffer, 0, offset);
                offset = 0;
            }
        }
        free(temp);
    }
    fputs(buffer, outfile);

    TIMER_STOP();
    printf("Transactions Classification Completed. Took %f seconds\n",time_delta);
    printf("Transactions per second: %f\n", tv->size /time_delta);

    fclose(outfile);
    destroyTransactionvector(tv);
}

typedef struct
{
    int threadID;
    Classvector *cv;
    Transactionvector *tv;
    FILE *outfile;
} ThreadArgs;

void *threadindividualsolve(void *vargp)
{
    ThreadArgs *args = (ThreadArgs *)vargp;

    int start = (args->tv->size / CORES) * args->threadID;
    int end = args->threadID == CORES - 1 ? args->tv->size : (args->tv->size / CORES) * (args->threadID + 1);

    int currectSize = BUFSIZ;
    char *buffer = (char *)malloc(currectSize * sizeof(char));
    int offset = 0;

    for (int i = start; i < end; i++)
    {
        char *temp = transactionToString(args->tv->transactions[i]);
        for (int j = 0; j < args->cv->size; j++)
        {
            if (classification_matches(args->cv->classes[j], args->tv->transactions[i]))
            {
                sprintf(buffer + offset, "%s%d\n", temp, args->cv->classes[j]->type);
                offset += strlen(temp) + NumberLen(args->cv->classes[j]->type) + 1;
            }
            if (offset >= currectSize - 60)
            {
                currectSize *= 2;
                buffer = (char *)realloc(buffer, currectSize * sizeof(char));
            }
        }
        free(temp);
    }
    fputs(buffer, args->outfile);
    free(buffer);
    free(args);
    return NULL;
}

void parallelSolve_CPU(Solver *s, char *in, char *out)
{
    TIMER_START();
    Transactionvector *tv = create_transaction_vector();

    FILE *infile;
    infile = fopen(in, "r");

    char *line = NULL;
    ssize_t read = 0;
    size_t len = 0;
    while ((read = getline(&line, &len, infile)) > 0)
    {
        tv = addTransactionToVector(tv, create_transaction(line));
    }
    free(line);
    fclose(infile);

    TIMER_STOP();
    printf("Time to read Transactions = %f seconds\n",time_delta);

    FILE *outfile;
    outfile = fopen(out, "w");

    pthread_t ids[CORES];

    TIMER_START();

    for (int i = 0; i < CORES; i++)
    {
        ThreadArgs *args = (ThreadArgs *)malloc(sizeof(ThreadArgs));
        args->threadID = i;
        args->cv = s->vector;
        args->tv = tv;
        args->outfile = outfile;
        pthread_create(&ids[i], NULL, threadindividualsolve, (void *)args);
    }

    for (int i = 0; i < CORES; i++)
    {
        pthread_join(ids[i], NULL);
    }

    TIMER_STOP();
    printf("Transactions Classification Completed. Took %f seconds\n",time_delta);
    printf("Parallel Transactions per second: %f\n", tv->size /time_delta);

    fclose(outfile);
    destroyTransactionvector(tv);
}

void destroySolver(Solver *s)
{
    destroyClassvector(s->vector);
    free(s);
}

__global__ void call_GPU(int **matrix, int *classPosition, int *transactions, int **output, int trans_size)
{
    int t = (blockIdx.x * blockDim.x + threadIdx.x); // transaction index
    int c = 0;                                       // classification index
    int r = 0;                                       // rule index
    short matches = 1;
    int classCount = 0;

    while (t < trans_size)
    {
        while (matrix[c] != NULL)
        {
            while (matrix[c][r * TUPLESIZE] != -9)
            {
                matches = 1;
                for (int k = 0; k < TUPLESIZE; k++)
                {
                    int value = matrix[c][r * TUPLESIZE + k];
                    if (transactions[t * TUPLESIZE + k] != value && value != -1)
                    {
                        matches = 0;
                        break;
                    }
                }
                if (matches)
                {
                    output[t][classCount] = classPosition[c];
                    classCount++;
                    break;
                }
                r++;
            }
            c++;
            r = 0;
        }
        if (classCount != 101)
        {
            output[t][classCount] = -1;
        }
        t += blockDim.x * gridDim.x;
        c = 0;
        classCount = 0;
    }
}

void parallelSolve(Solver *s, char *in, char *out)
{
    TIMER_START();

    Transactionvector *tv = create_transaction_vector();

    FILE *infile;
    infile = fopen(in, "r");

    char *line = NULL;
    ssize_t read = 0;
    size_t len = 0;
    while ((read = getline(&line, &len, infile)) > 0)
    {
        tv = addTransactionToVector(tv, create_transaction(line));
    }
    free(line);
    fclose(infile);

    TIMER_STOP();
    printf("Time to read Transactions = %f seconds\n",time_delta);

    TIMER_START();

    int **matrix;
    int *classPosition;
    int *transactions;
    int **output;
    int *array;

    HANDLE_ERROR(cudaMalloc(&classPosition, (s->vector->size + 1) * sizeof(int)));
    int n = -9;
    cudaMemcpy(&classPosition[s->vector->size], &n, sizeof(int), cudaMemcpyHostToDevice);
    HANDLE_ERROR(cudaMalloc(&matrix, (s->vector->size + 1) * sizeof(int *)));
    int *p = NULL;
    cudaMemcpy(&(matrix[s->vector->size]), &p, sizeof(int *), cudaMemcpyHostToDevice);

    for (int i = 0; i < s->vector->size; i++)
    {
        cudaMemcpy(&classPosition[i], &(s->vector->classes[i]->type), sizeof(int), cudaMemcpyHostToDevice);
        HANDLE_ERROR(cudaMalloc(&array, s->vector->classes[i]->vector->size * sizeof(int) * TUPLESIZE + sizeof(int)));
        p = array + s->vector->classes[i]->vector->size * TUPLESIZE;
        cudaMemcpy(p, &n, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&matrix[i], &array, sizeof(int *), cudaMemcpyHostToDevice);

        for (int j = 0; j < s->vector->classes[i]->vector->size; j++)
        {
            int *temp = s->vector->classes[i]->vector->rules[j].tuple;
            p = array + j * TUPLESIZE;
            cudaMemcpy(p, temp, sizeof(int) * TUPLESIZE, cudaMemcpyHostToDevice);
        }
    }

    HANDLE_ERROR(cudaMalloc(&transactions, tv->size * sizeof(int) * TUPLESIZE));
    HANDLE_ERROR(cudaMalloc(&output, tv->size * sizeof(int *)));

    for (int i = 0; i < tv->size; i++)
    {
        p = transactions + i * TUPLESIZE;
        cudaMemcpy(p, tv->transactions[i].tuple, sizeof(int) * TUPLESIZE, cudaMemcpyHostToDevice);
        HANDLE_ERROR(cudaMalloc(&array, 101 * sizeof(int)));
        cudaMemcpy(&output[i], &array, sizeof(int *), cudaMemcpyHostToDevice);
    }
    TIMER_STOP();

    printf("Memory alocated in GPU. Took %f seconds\n",time_delta);

    printf("Kernel Call.\n");

    TIMER_START();

    call_GPU<<<24, 256>>>(matrix, classPosition, transactions, output, tv->size);
    printf("Kernel finished.\n");

    TIMER_STOP();
    printf("Transactions Classification Completed. Took %f seconds\n",time_delta);
    printf("Transactions per second: %f\n",time_delta ? tv->size /time_delta : tv->size);

    /////////////////////////////////////////////////////////// write to file

    TIMER_START();

    FILE *outfile;
    outfile = fopen(out, "w");

    char buffer[BUFSIZ] = {0};
    int offset = 0;
    int *classes = (int *)malloc(101 * sizeof(int));
    int j = 0;

    for (int i = 0; i < tv->size; i++)
    {
        cudaMemcpy(&p, &output[i], sizeof(int *), cudaMemcpyDeviceToHost);
        cudaMemcpy(classes, p, sizeof(int) * 101, cudaMemcpyDeviceToHost);
        char *temp = transactionToString(tv->transactions[i]);
        j = 0;
        while (j < 101 && classes[j] != -1)
        {
            sprintf(buffer + offset, "%s%d\n", temp, classes[j]);
            offset += strlen(temp) + NumberLen(classes[j]) + 1;
            if (offset >= BUFSIZ - 60)
            {
                fputs(buffer, outfile);
                memset(buffer, 0, offset);
                offset = 0;
            }
            j++;
        }
        free(temp);
    }
    fputs(buffer, outfile);

    free(classes);

    TIMER_STOP();
    printf("Resulte in File. Took %f seconds\n",time_delta);

    /////////////////////////////////////////////////////////////////////

    TIMER_START();
    for (int i = 0; i < s->vector->size; i++)
    {
        cudaMemcpy(&p, &matrix[i], sizeof(int *), cudaMemcpyDeviceToHost);
        cudaFree(p);
    }

    for (int i = 0; i < tv->size; i++)
    {
        cudaMemcpy(&p, &output[i], sizeof(int *), cudaMemcpyDeviceToHost);
        cudaFree(p);
    }

    cudaFree(matrix);
    cudaFree(classPosition);
    cudaFree(transactions);
    cudaFree(output);

    TIMER_STOP();
    printf("GPU memory freed. Took %f seconds\n",time_delta);
    destroyTransactionvector(tv);
}

int main()
{
    //Solver *s = create_solver("../../Input/rule_tiny.csv");
    // solve(s, "../../Input/transactions_tiny.csv", "../../Output/transactions_tiny_output.csv");

    Solver *s = create_solver("../../Input/rule_2M.csv");
    //solve(s, "../../Input/transactions_0.csv", "../../Output/transactions_0_output.csv");
    //parallelSolve_CPU(s, "../../Input/transactions_0.csv", "../../Output/transactions_0_output.csv");
    parallelSolve(s, "../../Input/transactions_0.csv", "../../Output/transactions_0_output.csv");

    destroySolver(s);
    return 0;
}
