/* Parallel Programming Language Semantics Coursework 2
 * s0925570@sms.ed.ac.uk */

/* BEGIN PROVIDED CODE */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "stack.h"

#define EPSILON 1e-3
#define F(arg)  cosh(arg)*cosh(arg)*cosh(arg)*cosh(arg)
#define A 0.0
#define B 5.0

#define SLEEPTIME 1

int *tasks_per_process;

double farmer(const int);

void worker(const int);

int
main(int argc, char **argv) {
  int i, myid, numprocs;
  double area, a, b;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  if(numprocs < 2) {
    fprintf(stderr, "ERROR: Must have at least 2 processes to run\n");
    MPI_Finalize();
    exit(1);
  }

  if (myid == 0) { // Farmer
    // init counters
    tasks_per_process = (int *) malloc(sizeof(int)*(numprocs));
    for (i=0; i<numprocs; i++) {
      tasks_per_process[i]=0;
    }
  }

  if (myid == 0) { // Farmer
    area = farmer(numprocs);
  } else { //Workers
    worker(myid);
  }

  if (myid == 0) {
    fprintf(stdout, "Area=%lf\n", area);
    fprintf(stdout, "\nTasks Per Process\n");
    for (i=0; i<numprocs; i++) {
      fprintf(stdout, "%d\t", i);
    }
    fprintf(stdout, "\n");
    for (i=0; i<numprocs; i++) {
      fprintf(stdout, "%d\t", tasks_per_process[i]);
    }
    fprintf(stdout, "\n");
    free(tasks_per_process);
  }
  MPI_Finalize();
  return 0;
}

/* END PROVIDED CODE */

#define WORK_TAG 0
#define DIE_TAG 1

#define ACCURATE_TAG 0
#define INACCURATE_TAG 1

void
push_endpoints_onto_stack(const double start_point, const double end_point, stack *stack) {
  double boundaries[2] =  { start_point, end_point };
  push(boundaries, stack);
}

int
distribute_stack_to_workers(stack *stack, const int numprocs) {
  int worker_number = 0;
  int workers_with_tasks = 0;

  while (++worker_number < numprocs) {
    if (is_empty(stack)) break;

    double start, end, *head;
    head = pop(stack);
    start = head[0]; end = head[1];

    double params[2] = { start, end };

    workers_with_tasks++;
    tasks_per_process[worker_number]++;
    MPI_Send(params, 2, MPI_DOUBLE, worker_number, WORK_TAG, MPI_COMM_WORLD);
  }

  return workers_with_tasks;
}

double
process_worker_responses(stack *stack, const int workers_with_tasks) {
  double workers_area = 0;
  int worker_number = 0;

  while (++worker_number <= workers_with_tasks) {
    MPI_Status status;
    double buffer[2];
    MPI_Recv(buffer, 2, MPI_DOUBLE, worker_number, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    // Accurate segment results increment total area.
    if (status.MPI_TAG == ACCURATE_TAG) {
      double segment_area = buffer[0];
      workers_area += segment_area;

    // Inaccurate segment results split segment in two for next iteratation of job distribution.
    } else {
      double start, midpoint, end;
      start = buffer[0]; end = buffer[1];
      midpoint = (start + end) / 2;

      push_endpoints_onto_stack(start, midpoint, stack);
      push_endpoints_onto_stack(midpoint, end, stack);
    }
  }

  return workers_area;
}

void
release_all_workers(const int numprocs) {
  int worker_number = 0;

  while (++worker_number < numprocs) {
    double params[2] = { 0., 0. };
    MPI_Send(params, 2, MPI_DOUBLE, worker_number, DIE_TAG, MPI_COMM_WORLD);
  }
}

double
farmer(const int numprocs) {
  double total_area = 0;

  stack *stack = new_stack();
  push_endpoints_onto_stack(A, B, stack);

  while (!is_empty(stack)) {
    int workers_with_tasks = distribute_stack_to_workers(stack, numprocs);

    double new_area = process_worker_responses(stack, workers_with_tasks);
    total_area += new_area;
  }
  release_all_workers(numprocs);

  return total_area;
}

void
worker(const int mypid) {
  while (1) {
    double params[2];
    double start, mid, end, f_start, f_mid, f_end, estimate, larea, rarea;
    MPI_Status status;

    MPI_Recv(params, 2, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    if (status.MPI_TAG == WORK_TAG) {
      start = params[0]; end = params[1];
      f_start = F(start); f_end = F(end);
      estimate = (f_start + f_end) * (end - start) / 2;

      mid = (start + end) / 2;
      f_mid = F(mid);
      larea = (f_start + f_mid) * (mid - start) / 2;
      rarea = (f_mid + f_end) * (end - mid) / 2;

      double segment_area = larea + rarea;

      usleep(SLEEPTIME);

      int accurate_enough = fabs((segment_area) - estimate) <= EPSILON;

      if (accurate_enough) {
        params[0] = segment_area;
        MPI_Send(params, 2, MPI_DOUBLE, 0, ACCURATE_TAG, MPI_COMM_WORLD);
      } else {
        params[0] = start;
        params[1] = end;
        MPI_Send(params, 2, MPI_DOUBLE, 0, INACCURATE_TAG, MPI_COMM_WORLD);
      }
    } else {
      break;
    }
  }
}

/* Description

Each layer of the stack is used to keep track of the start and endpoints of each segment needing to be computed.
Initially the range between the start and the endpoint are pushed onto the stack.

The farmer loops through the process of popping regions off of the stack and distributing them to worker processes
until either the stack is empty or it runs out of workers.
Giving the worker a region is done by utilising MPI_Send with two MPI_DOUBLEs representing the start and end point to be calculated.
The estimates and other calculations are all done on the worker in order to leave the farmer with as little work
in the runtime loop as possible to improve the rate of job distribution.
The WORK_TAG MPI_TAG is used to inform that worker that it should attempt to compute the area and then return its result.

The farmer then recieves results from all utilised workers by using MPI_Recv once for each worker that has a task.
If the worker responds with an ACCURATE_TAG MPI_TAG, the farmer adds its calculated MPI_DOUBLE segment area to the total.
If the worker responds with an INACCURATE_TAG MPI_TAG, the farmer reads the attempted start and endpoint MPI_DOUBLEs
then bisects the range and pushes it back onto the stack.
This runtime loop continues until the stack is empty after all workers have been queried, this means that there must be
an accurate result for each segment initially on the stack and the sum will be the total integral.
When the area has been computed, the farmer sends a DIE_TAG MPI_TAG broadcast to each worker to allow them to exit their runtime loop.

The worker processes loop calling MPI_Recv and then attempting to compute the area between the two MPI_DOUBLE endpoints they receive.
If the result is accurate, they send back an MPI_DOUBLE with the area marked with the ACCURATE_TAG.
If the result is innacurate, they send back the endpoints they received initially with the INACCURATE_TAG.
This loop continues until the worker receive work tagged with a DIE_TAG MPI_TAG, at which point they exit.

*/
