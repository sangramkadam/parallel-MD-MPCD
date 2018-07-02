/*----------------------------------------------------------------------
pmd.h is an include file for a parallel MD program, pmd.c.
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#define NINT(a) ((a) >= 0.0 ? (int)((a)+0.5) : (int)((a)-0.5))	//Nearest integer
#define Sqr(x)     ((x) * (x))
#define Cube(x)    ((x) * (x) * (x))
#define AllocMem(a, n, t)  a = (t *) malloc ((n) * sizeof (t))
/* Constants------------------------------------------------------------
----------------------------------------------------------------------*/
#define L 30.0
#define vproc 2
#define rho 10.0
#define temp 1.0
#define dt_md 0.001
#define dt_mpc 0.1
#define epsA 1.0
#define epsB 0.1
#define epsPolymer 2.0
#define sigma_sc 1.0
#define sigma_cc 2.0
#define alpha 1.57079632679
#define stepLimit 10
#define stepAvg 100000
#define ks 30.0
#define kappa 10.0
#define ms 1.0
#define mc rho*4.188790205*Cube(sigma_sc)
#define Ncol 8
#define Nsol NINT(rho*(Cube(L)-mc*((double)Ncol)/rho)/ms)
#define nproc Cube(vproc)
#define MOVED_OUT -1.0e10
#define EMPTY -1
/* Constants for the random number generator */
#define D2P31M 2147483647.0
#define DMUL 16807.0

/* Variables------------------------------------------------------------

box[0|1|2] = Box length per processor in the x|y|z direction.
n = # of resident atoms in this processor.
nb = # of copied boundary atoms from neighbors.
N = Total # of atoms summed over processors.
r[NEMAX][3]: r[i][0|1|2] is the x|y|z coordinate of atom i (including 
  the copied atoms).
rv[NEMAX][3]: rv[i][0|1|2] is the x|y|z velocity of atom i (including 
  the copied atoms).
ra[NMAX][3]: ra[i][0|1|2] is the x|y|z acceleration on atom i.
dbuf[NDBUF]: Buffer for sending double-precision data
dbufr[NDBUF]:           receiving
vproc[0|1|2] = # of processors in the x|y|z direction.
nproc = # of processors = vproc[0]*vproc[1]*vproc[2].
sid = Sequential processor ID.
vid[3] = Vector processor ID;
  sid = vid[0]*vproc[1]*vproc[2] + vid[1]*vproc[2] + vid[2].
NN[6]: NN[ku] is the node ID of the neighbor specified by a neighbor.
  index, ku.  The neighbor index is defined as:
  ku = 0: xlow  (West );
       1: xhigh (East );
       2: ylow  (South);
       3: yhigh (North);
       4: zlow  (Down );
       5: zhigh (Up   ).
sv[6][3]: sv[ku][] is the shift vector to the ku-th neighbor.
myparity[0|1|2] = Parity of vector processor ID in the x|y|z direction.
lsb[6][NBMAX]: lsb[ku][0] is the total # of boundary atoms to be sent
  to neighbor ku; lsb[ku][k] is the atom ID, used in r, of the k-th 
  atom to be sent.
status: Returned by MPI message-passing routines.
cpu: Elapsed wall-clock time in seconds.
comt: Communication time in seconds.
lc[3]: lc[0|1|2] is the # of cells in the x|y|z direction.
rc[3]: rc[0|1|2] is the length of a cell in the x|y|z direction.
lscl[NEMAX]: Linked cell lists.
head[NCLMAX]: Headers for the linked cell lists.
kinEnergy = Kinetic energy.
potEnergy = Potential energy.
totEnergy = Total energy.
temperature = Current temperature.
stepCount = Current time step.
----------------------------------------------------------------------*/
double t;
double box;
int master;
int ns,nc,nbCol,nbSol,nsA,nsB;
int ncMax,nsMax;
int nbColMax,nbSolMax;
double **rs,**vs,**fs;
double **rc,**vc,**fc,**grc;
int *colID,*bColID;
int *solventFlag,*monomerFlag;
double *dbuf,*dbufr;
int ibuf[nproc];
int sid,vid[3],nn[6],myparity[3],world_size;
int *lsbSol[6],*lsbCol[6];
MPI_Status status;
MPI_Comm useProc;
double cpu,comt;
//int head[NCLMAX],lscl[NEMAX],lc[3];
//double rc[3];


double kinEnergy,potEnergy,totEnergy,temperature;
int stepCount;
double Uc, Duc;    /* Potential cut-off parameters */

double lStart[3],lEnd[3]; //start and end of box length
double rc_sc,rc_cc;
double rc2_sc,rc2_cc;
double rc6_sc,rc6_cc,sigma6_sc,sigma6_cc;
double ecut_sc,ecut_cc;
double r_eq;
/* Functions & function prototypes------------------------------------*/

double SignR(double v,double x) {if (x > 0) return v; else return -v;}
double Dmod(double a, double b) {
  int n;
  n = (int) (a/b);
  return (a - b*n);
}
double RandR(double *seed) {
  *seed = Dmod(*seed*DMUL,D2P31M);
  return (*seed/D2P31M);
}
void RandVec3(double *p, double *seed) {
  double x,y,s = 2.0;
  while (s > 1.0) {
    x = 2.0*RandR(seed) - 1.0; y = 2.0*RandR(seed) - 1.0; s = x*x + y*y;
  }
  p[2] = 1.0 - 2.0*s; s = 2.0*sqrt(1.0 - s); p[0] = s*x; p[1] = s*y;
}

void Initialization(void);
void InitMPI(void);
void InitParameters();
void InitNeighProc();
void InitConfiguration();
void FreeMemory();
int FindProc(double,double,double);
int IsBoundaryAtom(int i, int ku);
void AtomCopy();
void CreateNighborsList(int ncell,int nc_dim,int neigh[][27]);
void CreateLinkedList(int hoc[],int ll[],double a,int n,int ncell,int nc_dim);
void SingleMPCDstep(void);
void SingleMDstep(void);
void VerletUpdate1(void);
void VerletUpdate2(void);
void PeriodicBoundary(double* x,double*y,double*z);
void Conversion(int id);
void BackConversion(void);
void AtomMove();
int IsMovedAtom(int i,int ku);
void Sample(double *ke,double* sumv);
void eval_props();
int bbd(double* ri, int ku);
int bmv(double* ri, int ku);
/*--------------------------------------------------------------------*/


