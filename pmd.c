/*----------------------------------------------------------------------
Program pmd.c performs parallel molecular-dynamics for Lennard-Jones 
systems using the Message Passing Interface (MPI) standard.
----------------------------------------------------------------------*/
#include "pmd.h"

/*--------------------------------------------------------------------*/
int main(int argc, char **argv) 
{
int i;
double cpu1;
FILE *fp;
MPI_Init(&argc,&argv); // Initialize the MPI environment 
MPI_Comm_rank(MPI_COMM_WORLD, &sid);  // My processor ID 
MPI_Comm_size(MPI_COMM_WORLD,&world_size);
int color = sid/Cube(vproc);
// Split the communicator based on the color and use the
// original rank for ordering
MPI_Comm_split(MPI_COMM_WORLD, color, sid, &useProc);
  if(sid==0)
  {
    if(world_size<nproc)
    {
    fprintf(stderr,"\nERROR : Inadequate number of processors\n\n");
    MPI_Abort(MPI_COMM_WORLD,1);
    }
  }
  if(sid<nproc)
  {  
  Initialization(); 
    while(stepCount<stepLimit)
    {
    SingleMDstep();
    t = stepCount * dt_md;
    stepCount++;

        if(stepCount%1000==0)
        {
        sprintf(file,"pos.xyz");
        fp=fopen(file,"a");
        fprintf(fp,"%d\n%d\n",np_c,stepCount/1000);
        for(i=0;i<nc;i++)
        fprintf(fp,"%d %lf %lf %lf\n",0,rc[i][0],rc[i][1],rc[i][2]);
        fclose(fp);
        }

    }

  for(i=0;i<nc;i++)
  printf("%d\t%d %lf %lf %lf\n",sid,colID[i],fc[i][0],fc[i][1],fc[i][2]);


  }

MPI_Finalize(); // Clean up the MPI environment 
return 0;
}


/*---------------------------------------------------------------------
-----------------------------------------------------------------------
FUNCTION DEFINITIONS
-----------------------------------------------------------------------
---------------------------------------------------------------------*/

void Initialization()
{
InitParameters();
InitNeighProc();
InitConfiguration();
}


/*--------------------------------------------------------------------*/
void InitParameters()
{
/*----------------------------------------------------------------------
Initializes parameters.
----------------------------------------------------------------------*/
  int a,i;
  FILE *fp;

  master = (sid==0);
  // Vector index of this processor 
  vid[0] = sid/(vproc*vproc);
  vid[1] = (sid/vproc)%vproc;
  vid[2] = sid%vproc;

  rc_sc=pow(2.0,1.0/6.0)*sigma_sc;
  rc_cc=pow(2.0,1.0/6.0)*sigma_cc;
  rc2_sc=Sqr(rc_sc);
  rc2_cc=Sqr(rc_cc);
  sigma6_sc=Sqr(Cube(sigma_sc));
  sigma6_cc=Sqr(Cube(sigma_cc));
  rc6_sc=(sigma6_sc/(Cube(rc2_sc)));
  rc6_cc=(sigma6_cc/(Cube(rc2_cc)));
  ecut_sc=4.0*epsA*rc6_sc*(rc6_sc-1.0);
  ecut_cc=4.0*epsA*rc6_cc*(rc6_cc-1.0);
  r_eq = rc_cc+0.1;
  //box size of subregions belonging to each processor
  box = L/((double)vproc);
  //Find start length and end length in each dimension


  lStart[0] = box*vid[0];
  lStart[1] = box*vid[1];
  lStart[2] = box*vid[2];
  lEnd[0] = box*(vid[0]+1);
  lEnd[1] = box*(vid[1]+1);
  lEnd[2] = box*(vid[2]+1); 
  ncMax=Ncol;
  nsMax=(2*Nsol)/nproc;
  nbColMax=Ncol;
  nbSolMax=nsMax;


    AllocMem(colID,ncMax,int);
    AllocMem(solventFlag,nsMax,int);
    AllocMem(monomerFlag,ncMax,int);
    AllocMem(rc,ncMax,double*);
    AllocMem(grc,ncMax,double*);
    AllocMem(vc,ncMax,double*);
    AllocMem(fc,ncMax,double*);
    AllocMem(rs,nsMax,double*);
    AllocMem(vs,nsMax,double*);
    AllocMem(fs,nsMax,double*);
    for(i=0;i<ncMax;i++)
    {
    AllocMem(rc[i],3,double);
    AllocMem(grc[i],3,double);
    AllocMem(vc[i],3,double);
    AllocMem(fc[i],3,double);
    }
    for(i=0;i<nsMax;i++)
    {
    AllocMem(rs[i],3,double);
    AllocMem(vs[i],3,double);
    AllocMem(fs[i],3,double);
    }
    for(i=0;i<6;i++)
    {
    AllocMem(lsbCol[i],nbColMax,int);
    AllocMem(lsbSol[i],nbSolMax,int);
    }
    AllocMem(dbuf,4*nbSolMax+4*nbColMax,double);
    AllocMem(dbufr,4*nbSolMax+4*nbColMax,double);
    

  if(master)
  {
  printf("\n------------------------------------\n");
  printf("\n---Reading Simulation Parameters---\n");
  printf("\n------------------------------------\n\n");
  printf("Number of colloidal particles \t: %d\n",Ncol);
  printf("Box dimenstions \t\t: %lf\n",L);
  printf("Processors per dimension \t: %d\n",vproc);
  printf("Density \t\t\t: %lf\n",rho);
  printf("Initial Temperature \t\t: %lf\n",temp);
  printf("MD and MPCD time step \t\t: %lf %lf\n",dt_md,dt_mpc);
  printf("Energy parameter for A,B and Polymer :%lf %lf %lf\n",epsA,epsB,epsPolymer);
  printf("Length parameter \t\t: %lf %lf\n",sigma_sc,sigma_cc);
  printf("MPCD Cell rotation angle \t: %lf\n",alpha);
  printf("Step Limit \t\t\t: %d\n",stepLimit);
  printf("Step Avg \t\t\t: %d\n",stepAvg);
  printf("Spring const & 3-body force const: %lf %lf\n",ks,kappa);
  }
}

/*--------------------------------------------------------------------*/
void InitNeighProc() {
/*----------------------------------------------------------------------
Defines a logical network topology.  Prepares a neighbor-node ID table, 
nn, & a shift-vector table, sv, for internode message passing.  Also 
prepares the node parity table, myparity.
----------------------------------------------------------------------*/
  // Integer vectors to specify the six neighbor nodes 
  int iv[6][3] = {
    {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}
  };
  int ku,a,k1[3];

  // Set up neighbor tables, nn & sv 
  for (ku=0; ku<6; ku++) {
    // Vector index of neighbor ku 
    for (a=0; a<3; a++)
      k1[a] = (vid[a]+iv[ku][a]+vproc)%vproc;
    // Scalar neighbor ID, nn 
    nn[ku] = k1[0]*Sqr(vproc)+k1[1]*vproc+k1[2];

  }

  // Set up the node parity table, myparity 
  for (a=0; a<3; a++) {
    if (vproc == 1) 
      myparity[a] = 2;
    else if (vid[a]%2 == 0)
      myparity[a] = 0;
    else
      myparity[a] = 1;
  }
}



void InitConfiguration() 
{
int i,j,proc,id;
double rx,ry,rz,vx,vy,vz;
FILE *fp;
//Initialize number of particles in each processor to zero
nc=0;
ns=0;
fp = fopen("col.dat","r");
    for(i=0;i<Ncol;i++)
    {
    fscanf(fp,"%d%lf%lf%lf%lf%lf%lf",&id,&rx,&ry,&rz,&vx,&vy,&vz);
    proc = FindProc(rx,ry,rz);
        if(sid==proc)
        {
        colID[nc] = id;
        rc[nc][0] = rx;
        rc[nc][1] = ry;
        rc[nc][2] = rz;
        vc[nc][0] = vx;
        vc[nc][1] = vy;
        vc[nc][2] = vz;
        nc++;
        }
    }
fclose(fp);

fp = fopen("sol.dat","r");
    for(i=0;i<Nsol;i++)
    {
    fscanf(fp,"%d%lf%lf%lf%lf%lf%lf",&id,&rx,&ry,&rz,&vx,&vy,&vz);
    proc = FindProc(rx,ry,rz);
        if(sid==proc)
        {
        solventFlag[ns]=id;
        rs[ns][0] = rx;
        rs[ns][1] = ry;
        rs[ns][2] = rz;
        vs[ns][0] = vx;
        vs[ns][1] = vy;
        vs[ns][2] = vz;
        ns++;
        }
    }
fclose(fp);

    for(i=0;i<ncMax;i++)
    {                       //flag 0 catalytic
    monomerFlag[i]=0;       //flag 1 non catalytic
    }                       //flag 2 inactive
//Initialize global positions of colloids
  for(i=0;i<nc;i++)
    for(j=0;j<3;j++)
    grc[i][j]=rc[i][j];
//Initialize forces
  for(i=0;i<ns;i++)
  {
    for(j=0;j<3;j++)
    fs[i][j]=0.0;
  }
  for(i=0;i<nc;i++)
  {
    for(j=0;j<3;j++)
    fc[i][j]=0.0;
  }

}


/*------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------
Finds out the processor in which given prticle belongs
--------------------------------------------------------------------------------*/
int FindProc(double xx,double yy,double zz)
{
int px,py,pz,pp;
px = (int)(xx/box);
py = (int)(yy/box);
pz = (int)(zz/box);
pp = px*Sqr(vproc) + py*vproc + pz;
return pp;
}


/*--------------------------------------------------------------------*/
/*----------------------------------------------------------------------
Exchanges boundary-atom coordinates among neighbor nodes:  Makes 
boundary-atom list, LSB, then sends & receives boundary atoms.
----------------------------------------------------------------------*/

void AtomCopy() 
{
  int kd,kdd,i,j,k,ku,inode,nsd,nrc,a;
  int nbNewSol = 0,nbNewCol = 0; // # of "received" boundary atoms 
  double com1;

// Main loop over x, y & z directions starts--------------------------

  for (kd=0; kd<3; kd++) 
  {

    // Make a boundary-atom list, LSB---------------------------------

    // Reset the # of to-be-copied atoms for lower&higher directions 
    for (kdd=0; kdd<2; kdd++) 
    {
    lsbSol[2*kd+kdd][0] = 0;
    }

    // Scan all the residents & copies to identify boundary solvent particles  
    for (i=0; i<ns+nbNewSol; i++) 
    {
      for (kdd=0; kdd<2; kdd++) 
      {
        ku = 2*kd+kdd; // Neighbor ID 
        // Add an atom to the boundary-atom list, LSB, for neighbor ku 
        // according to bit-condition function, bbd 
        if (IsBoundaryAtom(i,ku)) lsbSol[ku][++(lsbSol[ku][0])] = i;
      }

    }

    // Message passing------------------------------------------------

    com1=MPI_Wtime(); // To calculate the communication time 

    // Loop over the lower & higher directions 
    for (kdd=0; kdd<2; kdd++) 
    {
      inode = nn[ku=2*kd+kdd]; // Neighbor node ID 

      // Send & receive the # of boundary atoms-----------------------

      nsd = lsbSol[ku][0]; // # of atoms to be sent 
      // Even node: send & recv 
      if (myparity[kd] == 0) {
        MPI_Send(&nsd,1,MPI_INT,inode,10,MPI_COMM_WORLD);
        MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,10,
                 MPI_COMM_WORLD,&status);
      }
      // Odd node: recv & send 
      else if (myparity[kd] == 1) {
        MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,10,
                 MPI_COMM_WORLD,&status);
        MPI_Send(&nsd,1,MPI_INT,inode,10,MPI_COMM_WORLD);
      }
      // Single layer: Exchange information with myself 
      else
      {
      nrc = nsd;
      }
      // Now nrc is the # of atoms to be received 

      // Send & receive information on boundary atoms-----------------

      // Message buffering 
      for (i=1; i<=nsd; i++)
      {
        for (a=0; a<3; a++)
        {
          if(vid[kd]==0 && kdd==0 && a==kd) 
          dbuf[3*(i-1)+a] = rs[lsbSol[ku][i]][a]+L;
          else if((vid[kd]==vproc-1) && kdd==1 && a==kd) 
          dbuf[3*(i-1)+a] = rs[lsbSol[ku][i]][a]-L;
          else 
          dbuf[3*(i-1)+a] = rs[lsbSol[ku][i]][a]; 
        }
      }
      // Even node: send & recv 
      if (myparity[kd] == 0) {
        MPI_Send(dbuf,3*nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
        MPI_Recv(dbufr,3*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
                 MPI_COMM_WORLD,&status);
      }
      // Odd node: recv & send 
      else if (myparity[kd] == 1) {
        MPI_Recv(dbufr,3*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
                 MPI_COMM_WORLD,&status);
        MPI_Send(dbuf,3*nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
      }
      // Single layer: Exchange information with myself 
      else
        for (i=0; i<3*nrc; i++) 
        dbufr[i] = dbuf[i];
      // Message storing 

      for (i=0; i<nrc; i++)
      {
        for (a=0; a<3; a++) 
        rs[ns+nbNewSol+i][a] = dbufr[3*i+a];
      }
      // Increment the # of received boundary atoms 
      nbNewSol = nbNewSol+nrc;
      //printf("%d %d %d %d %d\n",sid,ku,nsd,nrc,nbNewSol);
      // Internode synchronization 
      MPI_Barrier(useProc);
    } // Endfor lower & higher directions, kdd 

    comt += MPI_Wtime()-com1; // Update communication time, COMT 

  } // Endfor x, y & z directions, kd 



MPI_Barrier(useProc);
MPI_Allgather(&nc,1,MPI_INT,ibuf,1,MPI_INT,useProc);

  for(i=0;i<nproc;i++)
  {
    if(nc!=0 && sid==i)
    { //Message buffering for colloidal particles
      for(j=0;j<nc;j++)
      {
      k = j*4;
      dbuf[k] = (double)colID[j];
        for(a=0;a<3;a++)
        {
        dbuf[k+a+1] = rc[j][a];
        }
      }
    }
    if(ibuf[i]!=0)
    {//Broadcast the colloidal information
    MPI_Barrier(useProc);
    MPI_Bcast(dbuf,4*ibuf[i],MPI_DOUBLE,i,useProc);
      //Message storing
      if(sid!=i)
      {
        for(j=0;j<ibuf[i];j++)
        {
        k = j*4;
        colID[nc+nbNewCol+j] = NINT(dbuf[k]);
          for(a=0;a<3;a++)
          {
          rc[nc+nbNewCol+j][a] = dbuf[k+a+1];
          }
        }
      nbNewCol = nbNewCol+ibuf[i];
      }
    }
  }

  // Update the # of received boundary atoms 
  nbSol = nbNewSol;
  nbCol = nbNewCol;

}



/*--------------------------------------------------------------------*/
/*----------------------------------------------------------------------
Performs single MD step
----------------------------------------------------------------------*/

void SingleMDstep()
{
double xi,yi,zi;                            //position of i'th particle
double xr,yr,zr,r,r2,r2i,r6i;               //distance between particles
double en,ke,etot,sumv;
double eps,a;
int i,j,n,m,id,ff;
int idMin,idMax,istart,iend,IDseq[nc+nbCol];

VerletUpdate1();
AtomMove();
AtomCopy();

en=0.0;
etot=0.0;
ke=0.0;
  //Initialize all Forces to zero
  for(i=0;i<ns;i++)
  {
    for(j=0;j<3;j++)
    fs[i][j]=0.0;
  }
  for(i=0;i<nc;i++)
  {
    for(j=0;j<3;j++)
    fc[i][j]=0.0;
  }
  //Interaction of colloids with all the solvents
  for(n=0;n<nc+nbCol;n++)
  {
    for(id=0;id<ns+nbSol;id++)
  	{
    	//while(id!=-1)
    	//{
   		xr=rc[n][0]-rs[id][0];
   		yr=rc[n][1]-rs[id][1];
      zr=rc[n][2]-rs[id][2];
   		xr=xr-L*NINT(xr/L);
   		yr=yr-L*NINT(yr/L);
   		zr=zr-L*NINT(zr/L);	
   		r2=xr*xr+yr*yr+zr*zr;
   	    if(r2<rc2_sc)
   			{
        eps = epsA;
  	    r2i=(Sqr(sigma_sc))/r2;
        r6i=r2i*r2i*r2i;
        ff=eps*48.0*r2i*r6i*(r6i-0.5)/Sqr(sigma_sc);
          if(n<nc)
          {
          fc[n][0]=fc[n][0]+ff*xr;
          fc[n][1]=fc[n][1]+ff*yr;
          fc[n][2]=fc[n][2]+ff*zr;
          en=en+4.0*eps*r6i*(r6i-1.0)-ecut_sc;
          }
          if(id<ns)
          {
          fs[id][0]=fs[id][0]-ff*xr;
          fs[id][1]=fs[id][1]-ff*yr;
          fs[id][2]=fs[id][2]-ff*zr;
          }
  			}
   		//}
   	}
  }
  
  //Polymer polymer interaction
 
    for(i=0;i<nc+nbCol-1;i++)
    {    
    eps = epsPolymer;
    xi=rc[i][0];
    yi=rc[i][1];
    zi=rc[i][2];
      for(j=i+1;j<nc+nbCol;j++)
      {
      xr=xi-rc[j][0];
      yr=yi-rc[j][1];
      zr=zi-rc[j][2];
      xr=xr-L*NINT(xr/L);
      yr=yr-L*NINT(yr/L);
      zr=zr-L*NINT(zr/L);
      r2=xr*xr+yr*yr+zr*zr;
        if(r2<rc2_cc)          //???????? rc2 or rc2_cc
        {
        r2i=(sigma_cc*sigma_cc)/r2;
        r6i=r2i*r2i*r2i;
        ff=48.0*eps*r2i*r6i*(r6i-0.5)/Sqr(sigma_cc);
          if(i<nc)
          {
          fc[i][0]=fc[i][0]+ff*xr;
          fc[i][1]=fc[i][1]+ff*yr;
          fc[i][2]=fc[i][2]+ff*zr;
          en=en+4.0*eps*r6i*(r6i-1.0)-ecut_cc;
          }
          if(j<nc)
          {
          fc[j][0]=fc[j][0]-ff*xr;
          fc[j][1]=fc[j][1]-ff*yr;
          fc[j][2]=fc[j][2]-ff*zr;
          }
        }
      }
    }



VerletUpdate2();
//BackConversion();
//Sample(&ke,&sumv);
//en=en/((double)(np+np_c));
//etot=en+ke;

}



void VerletUpdate1(void)
{
int i,a;
double rold[3];
  for(i=0;i<ns;i++)
  {
    for(a=0;a<3;a++)
    {
    rs[i][a]=rs[i][a]+dt_md*vs[i][a]+0.5*Sqr(dt_md)*fs[i][a]/ms;
    vs[i][a]=vs[i][a]+0.5*dt_md*fs[i][a]/ms;
    }
    //PeriodicBoundary(&rs[i][0],&rs[i][1],&rs[i][2]);
  }

  for(i=0;i<nc;i++)
  {
    for(a=0;a<3;a++)
    {
    rold[a]=rc[i][a];
    rc[i][a]=rc[i][a]+dt_md*vc[i][a]+0.5*dt_md*dt_md*fc[i][a]/mc;
    vc[i][a]=vc[i][a]+0.5*dt_md*fc[i][a]/mc;
    grc[i][a]=grc[i][a]+(rc[i][a]-rold[a]);
    }
  PeriodicBoundary(&rc[i][0],&rc[i][1],&rc[i][2]);
  }

}


void VerletUpdate2(void)
{
int i,a;
  for(i=0;i<ns;i++)
  {
    for(a=0;a<3;a++)
    vs[i][a]=vs[i][a]+0.5*dt_md*fs[i][a]/ms;
  }
  for(i=0;i<nc;i++)
  {
    for(a=0;a<3;a++)
    vc[i][a]=vc[i][a]+0.5*dt_md*fc[i][a]/mc;
  }
}

void PeriodicBoundary(double* xx,double* yy,double* zz)
{
if(*xx<0.0)*xx=*xx+L;
if(*xx>L)*xx=*xx-L;
if(*yy<0.0)*yy=*yy+L;
if(*yy>L)*yy=*yy-L;
if(*zz<0.0)*zz=*zz+L;
if(*zz>L)*zz=*zz-L; 
}

void Conversion(int id)
{
double prob;
//prob = rand();
//prob = prob/RAND_MAX;

//    if(prob<0.5)
//    {
    nsB++;
    nsA--;
    solventFlag[id]=1;
//    }

}

void BackConversion(void)
{
int i;
double xr,yr,zr,r2;
double xc_cm,yc_cm,zc_cm;                   //centre of mass of polymer
xc_cm=0.0;
yc_cm=0.0;
zc_cm=0.0;
    for(i=0;i<nc+nbCol;i++)
    {
    xc_cm=xc_cm+rc[i][0];
    yc_cm=yc_cm+rc[i][1];
    zc_cm=zc_cm+rc[i][2];
    }
xc_cm=xc_cm/((double)(nc+nbCol));
yc_cm=yc_cm/((double)(nc+nbCol));
zc_cm=zc_cm/((double)(nc+nbCol));

    for(i=0;i<ns;i++)
    {
    if(solventFlag[i]!=0)
        {
        xr=rs[i][0]-xc_cm;
        xr=xr-L*NINT(xr/L);
        yr=rs[i][1]-yc_cm;
        yr=yr-L*NINT(yr/L);
        zr=rs[i][2]-zc_cm;
        zr=zr-L*NINT(zr/L);
        r2=xr*xr+yr*yr+zr*zr;
            if(r2>Sqr((L+L+L)/6))
            {
            solventFlag[i]=0;
            nsA++;
            nsB--;
            }
        }
    }
}


/*----------------------------------------------------------------------
Sends moved-out atoms to neighbor nodes and receives moved-in atoms 
from neighbor nodes.  Called with n, r[0:n-1] & rv[0:n-1], atom_move 
returns a new n' together with r[0:n'-1] & rv[0:n'-1].
----------------------------------------------------------------------*/

/* Local variables------------------------------------------------------

mvque[6][NBMAX]: mvque[ku][0] is the # of to-be-moved atoms to neighbor 
  ku; MVQUE[ku][k>0] is the atom ID, used in r, of the k-th atom to be
  moved.
----------------------------------------------------------------------*/

void AtomMove()
{
int mvque[6][nbSolMax];
int newim = 0; // # of new immigrants 
int ku,kd,i,j,k,proc,kdd,kul,kuh,inode,ipt,a,nsd,nrc;
double com1;

  //Reset the # of to-be-moved atoms, MVQUE[][0]
  for(ku=0; ku<6; ku++) mvque[ku][0] = 0;
  
  //Main loop over x, y & z directions starts------------------------
  for(kd=0; kd<3; kd++)
  {
  //Make a moved-atom list, mvque----------------------------------
  //Scan all the residents & immigrants to list moved-out atoms
    for(i=0; i<ns+newim; i++)
    {
    kul = 2*kd  ; // Neighbor ID 
    kuh = 2*kd+1;
      if(rs[i][0]>MOVED_OUT) //Don't scan moved-out atoms
      {
        //Move to the lower direction 
        if(IsMovedAtom(i,kul))
        mvque[kul][++(mvque[kul][0])] = i;
        //Move to the higher direction
        else if(IsMovedAtom(i,kuh))
        mvque[kuh][++(mvque[kuh][0])] = i;
      }
    }
    
  //Message passing with neighbor nodes----------------------------

  //Loop over the lower & higher directions------------------------
    for(kdd=0;kdd<2;kdd++)
    {
    inode = nn[ku=2*kd+kdd]; // Neighbor node ID
    
    //end & receive the # of boundary atoms---------------------------
    
    nsd = mvque[ku][0]; // # of atoms to-be-sent 
    //Even node: send & recv
      if (myparity[kd] == 0)
      {
      MPI_Send(&nsd,1,MPI_INT,inode,110,MPI_COMM_WORLD);
      MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,110,
               MPI_COMM_WORLD,&status);
      }
    //Odd node: recv & send
      else if (myparity[kd] == 1)
      {
      MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,110,
               MPI_COMM_WORLD,&status);
      MPI_Send(&nsd,1,MPI_INT,inode,110,MPI_COMM_WORLD);
      }
      //Single layer: Exchange information with myself
      else
      nrc = nsd;
      
      //Now nrc is the # of atoms to be received
      
      //Send & receive information on boundary atoms-----------------
      
      //Message buffering
      for (i=1; i<=nsd; i++)
      {
      PeriodicBoundary(&rs[mvque[ku][i]][0],&rs[mvque[ku][i]][1],&rs[mvque[ku][i]][2]);
        for (a=0; a<3; a++)
        {
        dbuf[6*(i-1)  +a] = rs[mvque[ku][i]][a];
        dbuf[6*(i-1)+3+a] = vs[mvque[ku][i]][a];
        rs[mvque[ku][i]][0] = MOVED_OUT; // Mark the moved-out atom 
        }
      }
      //Even node: send & recv, if not empty
      if (myparity[kd] == 0)
      {
      MPI_Send(dbuf,6*nsd,MPI_DOUBLE,inode,120,MPI_COMM_WORLD);
      MPI_Recv(dbufr,6*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,120,
               MPI_COMM_WORLD,&status);
      }
      //Odd node: recv & send, if not empty
      else if (myparity[kd] == 1)
      {
      MPI_Recv(dbufr,6*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,120,
               MPI_COMM_WORLD,&status);
      MPI_Send(dbuf,6*nsd,MPI_DOUBLE,inode,120,MPI_COMM_WORLD);
      }
      //Single layer: Exchange information with myself
      else
        for (i=0; i<6*nrc; i++) dbufr[i] = dbuf[i];
      
      //Message storing
      for (i=0; i<nrc; i++)
      {
        for (a=0; a<3; a++)
        {
        rs[ns+newim+i][a] = dbufr[6*i  +a];
        vs[ns+newim+i][a] = dbufr[6*i+3+a];
        }
      }
    //Increment the # of new immigrants
    newim = newim+nrc;
    
    //Internode synchronization
    MPI_Barrier(useProc);
    }//Endfor lower & higher directions, kdd

  }//Endfor x, y & z directions, kd 

//Compress resident arrays including new immigrants
ipt = 0;
  for (i=0; i<ns+newim; i++) 
  {
    if (rs[i][0] > MOVED_OUT) 
    {
      for (a=0; a<3; a++) {
      rs[ipt][a] = rs[i][a];
      vs[ipt][a] = vs[i][a];
      }
      ++ipt;
    }
  }
//Update the compressed # of resident atoms
ns = ipt;









newim = 0;
MPI_Barrier(useProc);
MPI_Allgather(&nc,1,MPI_INT,ibuf,1,MPI_INT,useProc);

  for(i=0;i<nproc;i++)
  {
    if(nc!=0 && sid==i)
    { //Message buffering for colloidal particles
      for(j=0;j<nc;j++)
      {
      k = j*7;
      dbuf[k] = (double)colID[j];
        for(a=0;a<3;a++)
        {
        dbuf[k+a+1] = rc[j][a];
        dbuf[k+a+4] = vc[j][a];
        }
      }
    }
    if(ibuf[i]!=0)
    {//Broadcast the colloidal information
    MPI_Barrier(useProc);
    MPI_Bcast(dbuf,7*ibuf[i],MPI_DOUBLE,i,useProc);
      //Message storing
      if(sid!=i)
      {
        for(j=0;j<ibuf[i];j++)
        {
        k = j*7;
        colID[nc+newim+j] = NINT(dbuf[k]);
          for(a=0;a<3;a++)
          {
          rc[nc+newim+j][a] = dbuf[k+a+1];
          vc[nc+newim+j][a] = dbuf[k+a+4];
          }
        }
      newim = newim+ibuf[i];
      }
    }
  }
  //Mark the moved-out atoms
  for(i=0;i<nc+newim;i++)
  {
  proc = FindProc(rc[i][0],rc[i][1],rc[i][2]);
    if(sid!=proc)
    rc[i][0] = MOVED_OUT;
  }
//Compress resident arrays including new immigrants
ipt = 0;
  for (i=0; i<nc+newim; i++) 
  {
    if (rc[i][0] > MOVED_OUT) 
    {
    colID[ipt] = colID[i];
      for (a=0; a<3; a++) 
      {
      rc[ipt][a] = rc[i][a];
      vc[ipt][a] = vc[i][a];
      }
      ++ipt;
    }
  }
//Update the compressed # of resident atoms
nc = ipt;




}




/*---------------------------------------------------------------------------
function is .true. if coordinate ri[3] is in the boundary to neighbor ku.
This function is only for solvent particles
---------------------------------------------------------------------------*/
int IsBoundaryAtom(int i, int ku)
{
int kd,kdd;
kd = ku/2; //x(0)|y(1)|z(2) direction
kdd = ku%2; // Lower(0)|higher(1) direction 
    if(kdd == 0)
    return rs[i][kd]-lStart[kd] < rc_sc;
    else
    return lEnd[kd]-rs[i][kd] < rc_sc;
}

int IsMovedAtom(int i,int ku)
{
int kd,kdd;
kd = ku/2; //x(0)|y(1)|z(2) direction
kdd = ku%2; // Lower(0)|higher(1) direction
    if(kdd == 0)
    return rs[i][kd]-lStart[kd] < 0.0;
    else
    return lEnd[kd]-rs[i][kd] < 0.0;
}
















