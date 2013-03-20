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

double quad (const double, const double, const double, const double, const double, int*);

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

void
distribute_endpoints_to_workers(const int numprocs) {
  double slice_width = (B - A) / (numprocs - 1);

  int worker_number = 0;
  while (++worker_number < numprocs) {
    double start, end;
    start = A + (worker_number - 1) * slice_width;
    end = A + worker_number * slice_width;

    double params[2] = { start, end };

    MPI_Send(params, 2, MPI_DOUBLE, worker_number, WORK_TAG, MPI_COMM_WORLD);
  }
}

double
process_worker_responses(const int numprocs) {
  double area = 0.;
  int worker_number = 0;

  while (++worker_number < numprocs) {
    double buffer[2];
    MPI_Recv(buffer, 2, MPI_DOUBLE, worker_number, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);

    double segment = buffer[0];
    area += segment;

    int tasks_needed = (int) buffer[1];
    tasks_per_process[worker_number] = tasks_needed;
  }

  return area;
}

double
farmer(const int numprocs) {
  double total_area = 0;

  distribute_endpoints_to_workers(numprocs);

  total_area = process_worker_responses(numprocs);

  return total_area;
}

void
worker(const int mypid) {
  double params[2];
  double start, mid, end, f_start, f_mid, f_end, estimate, larea, rarea;

  MPI_Recv(&params, 2, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);

  start = params[0]; end = params[1];
  f_start = F(start); f_end = F(end);
  estimate = (f_start + f_end) * (end - start) / 2;

  int tasks = 0;
  double segment_area = quad(start, end, f_start, f_end, estimate, &tasks);

  double buffer[2] = { segment_area, (double) tasks };

  MPI_Send(buffer, 2, MPI_DOUBLE, 0, WORK_TAG, MPI_COMM_WORLD);
}

/* Provided quad function with modification to track the number of calls */
double
quad( const double left, const double right,
      const double fleft, const double fright, const double lrarea, int *tasks) {
  double mid, fmid, larea, rarea;

  (*tasks)++;

  mid = (left + right) / 2;
  fmid = F(mid);
  larea = (fleft + fmid) * (mid - left) / 2;
  rarea = (fmid + fright) * (right - mid) / 2;
  if( fabs((larea + rarea) - lrarea) > EPSILON ) {
    larea = quad(left, mid, fleft, fmid, larea, tasks);
    rarea = quad(mid, right, fmid, fright, rarea, tasks);
  }
  return (larea + rarea);
}
