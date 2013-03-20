/* Parallel Programming Language Semantics Coursework 2
 * s0925570@sms.ed.ac.uk */

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

/* BEGIN PROVIDED CODE */

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

    double f_start = F(start);
    double f_end = F(end);

    double estimate = (f_start + f_end) * (end - start) / 2;

    double params[5] = { start, end, f_start, f_end, estimate };

    workers_with_tasks++;
    tasks_per_process[worker_number]++;
    MPI_Send(params, 5, MPI_DOUBLE, worker_number, WORK_TAG, MPI_COMM_WORLD);
  }

  return workers_with_tasks;
}

double
process_worker_responses(stack *stack, const int workers_with_tasks) {
  double workers_area = 0;
  int worker_number = 0;

  while (++worker_number <= workers_with_tasks) {
    int has_result;
    MPI_Recv(&has_result, 1, MPI_INT, worker_number, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);

    // Accurate segment results increment total area.
    if (has_result) {
      double segment_area;
      MPI_Recv(&segment_area, 1, MPI_DOUBLE, worker_number, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);

      workers_area += segment_area;

    // Inaccurate segment results split segment in two for next iteratation of job distribution.
    } else {
      double start, midpoint, end;
      double endpoints[2];
      MPI_Recv(&endpoints, 2, MPI_DOUBLE, worker_number, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
      start = endpoints[0]; end = endpoints[1]; midpoint = (start + end) / 2;

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
    double params[5] = { 0., 0., 0., 0., 0. };
    MPI_Send(params, 5, MPI_DOUBLE, worker_number, DIE_TAG, MPI_COMM_WORLD);
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
    double params[5];
    double start, mid, end, f_start, f_mid, f_end, estimate, larea, rarea;
    MPI_Status status;

    MPI_Recv(&params, 5, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    if (status.MPI_TAG == WORK_TAG) {
      start = params[0]; end = params[1]; f_start = params[2]; f_end = params[3]; estimate = params[4];

      mid = (start + end) / 2;
      f_mid = F(mid);
      larea = (f_start + f_mid) * (mid - start) / 2;
      rarea = (f_mid + f_end) * (end - mid) / 2;

      double segment_area = larea + rarea;

      int accurate_enough = fabs((segment_area) - estimate) <= EPSILON;

      MPI_Send(&accurate_enough, 1, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD);

      if (accurate_enough) {
        MPI_Send(&segment_area, 1, MPI_DOUBLE, 0, WORK_TAG, MPI_COMM_WORLD);
      } else {
        double endpoints[2] = { start, end };
        MPI_Send(endpoints, 2, MPI_DOUBLE, 0, WORK_TAG, MPI_COMM_WORLD);
      }
    } else {
      break;
    }
  }
}
