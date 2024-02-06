#ifndef BC_H
#define BC_H

/*
 * This file is part of the GraphBLAS Tutorial materials,
 * Copyright (c) 2018 Carnegie Mellon University and Intel Corporation.
 * All Rights Reserved
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND INTEL CORPORATION EXPRESSLY DISCLAIMS TO THE FULLEST
 * EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY WARRANTIES,
 * INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * Released under a BSD (SEI)-style license, please see LICENSE.txt for
 * full terms.
 *
 * DM18-xxx
 *
 * Authors: Scott McMillan, Timothy G. Mattson
 */

/**
 * @file betweenness_centrality.c
 *
 * @brief Two implementations of the betweenness centrality algorithm using
 *        the GraphBLAS C API. One for single source contribution and one for
 *        batch contribution.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <vector>
#include <stack>
#include <queue>
#include "GraphBLAS.h"
#include "../include/bwc_dijkstra.hpp"
using namespace std;

/* Source: "A Faster Algorithm for Betweenness Centrality", Brandes, 2001 */
int BC_brandes(float *bc, vector<int> *G, int *sources, int num_sources, int num_verts) {
  /* Initialize the BC vector with all zeros. */
  for (int i = 0; i < num_verts; i++) {
    bc[i] = 0.0;
  }

  for (int i = 0; i < num_sources; i++) {
    int s = sources[i];
    stack<int> S;
    vector<vector<int>> P(num_verts);
    vector<int> sigma(num_verts, 0);
    vector<int> d(num_verts, -1);
    queue<int> Q;

    sigma[s] = 1;
    d[s] = 0;
    Q.push(s);

    while (!Q.empty()) {
      int v = Q.front();
      Q.pop();
      S.push(v);

      for (int w = 0; w < num_verts; w++) {
        if (G[v][w] > 0) {
          if (d[w] < 0) {
            Q.push(w);
            d[w] = d[v] + 1;
          }

          if (d[w] == d[v] + 1) {
            sigma[w] = sigma[w] + sigma[v];
            P[w].push_back(v);
          }
        }
      }
    }

    vector<double> delta(num_verts, 0.0);
    while (!S.empty()) {
      int w = S.top();
      S.pop();
      for (int v : P[w]) {
        delta[v] = delta[v] + (sigma[v] / (double)sigma[w]) * (1 + delta[w]);
      }
      if (w != s) {
        bc[w] = bc[w] + delta[w];
      }
    }
  }

  return 0;
}

int BC_naive(float *bc, vector<int> *G, int *sources, int num_sources, int num_verts) {
  /* Initialize the BC vector with all zeros. */
  for (int i = 0; i < num_verts; i++) {
    bc[i] = 0.0;
  }

  vprop *assp[num_sources];
  for (int i = 0; i < num_sources; i++) {
    assp[i] = bwc_dijkstra(G, sources[i], num_verts);
  }

  int test = 1;

  vprop *ds;
  vprop *dt;
  for (int s = 0; s < num_sources; s++) {
    for (int t = 0; t < num_sources; t++) {
      if (sources[t] != sources[s]) {
        ds = assp[s];
        dt = assp[t];
        for (int v = 0; v < num_verts; v++) {
          if (v != sources[s] && v != sources[t] && ds[v].dist + dt[v].dist == ds[sources[t]].dist && ds[sources[t]].num_paths > 0) {
            bc[v] += (float)(ds[v].num_paths * dt[v].num_paths) / (float)ds[sources[t]].num_paths;
          }
        }
      }
    }
  }

  return 0;
}

//****************************************************************************
/*
 * Given a boolean n x n adjacency matrix A and a source vertex s,
 * compute the BC-metric vector delta, which should be empty on input.
 */
GrB_Info BC_GB(GrB_Vector *delta, GrB_Matrix A, GrB_Index s)
{
  GrB_Index n;
  GrB_Matrix_nrows(&n,A);                       // n = # of vertices in graph

  GrB_Vector_new(delta,GrB_FP32,n);             // Vector<float> delta(n)

  GrB_Matrix sigma;                             // Matrix<int32_t> sigma(n,n)
  GrB_Matrix_new(&sigma,GrB_INT32,n,n);         // sigma[d,k] = #shortest paths to node k at level d

  GrB_Vector q;
  GrB_Vector_new(&q, GrB_INT32, n);             // Vector<int32_t> q(n) of path counts
  GrB_Vector_setElement_INT32(q,1,s);           // q[s] = 1

  GrB_Vector p;                                 // Vector<int32_t> p(n) shortest path counts so far
  GrB_Vector_dup(&p, q);                        // p = q

  GrB_Monoid Int32Add;                          // Monoid <int32_t,+,0>
  GrB_Monoid_new_INT32(&Int32Add,GrB_PLUS_INT32,0);
  GrB_Semiring Int32AddMul;                     // Semiring <int32_t,int32_t,int32_t,+,*,0,1>
  GrB_Semiring_new(&Int32AddMul,Int32Add,GrB_TIMES_INT32);

  GrB_Descriptor desc;                          // Descriptor for vxm
  GrB_Descriptor_new(&desc);
  GrB_Descriptor_set(desc,GrB_MASK,GrB_COMP);   // structural complement of the mask
  GrB_Descriptor_set(desc,GrB_OUTP,GrB_REPLACE);// clear the output before assignment

  GrB_Descriptor tr1;                           // Transpose 1st input argument
  GrB_Descriptor_new(&tr1);
  GrB_Descriptor_set(tr1,GrB_INP0,GrB_TRAN);    // structural complement of the mask

  GrB_vxm(q,p,GrB_NULL,Int32AddMul,q,A,desc);   // get the first set of out neighbors

  /*
   * BFS phase
   */
  int32_t d = 0;                                // BFS level number
  int32_t sum = 0;                              // sum == include/GraphBLAS.h0 when BFS phase is complete
  do {
    GrB_Row_assign(sigma,GrB_NULL,GrB_NULL,q,d,GrB_ALL,n,GrB_NULL); // sigma[d,:] = q
    GrB_Vector_eWiseAdd_Semiring(p,GrB_NULL,GrB_NULL,Int32AddMul,p,q,GrB_NULL);     // accumulate path counts on this level
    GrB_vxm(q,p,GrB_NULL,Int32AddMul,q,A,desc);                     // q = # paths to nodes reachable
                                                                    //    from current level
    GrB_Vector_reduce_INT32(&sum,GrB_NULL,Int32Add,q,GrB_NULL);                  // sum path counts at this level
    ++d;
  } while (sum);

  /*
   * BC computation phase
   * (t1,t2,t3,t4) are temporary vectors
   */
  GrB_Monoid FP32Add;                           // Monoid <float,float,float,+,0.0>
  GrB_Monoid_new_FP32(&FP32Add,GrB_PLUS_FP32,0.0f);

  GrB_Monoid FP32Mul;                           // Monoid <float,float,float,*,1.0>
  GrB_Monoid_new_FP32(&FP32Mul,GrB_TIMES_FP32,1.0f);

  GrB_Semiring FP32AddMul;                      // Semiring <float,float,float,+,*,0.0,1.0>
  GrB_Semiring_new(&FP32AddMul,FP32Add,GrB_TIMES_FP32);

  GrB_Vector t1; GrB_Vector_new(&t1,GrB_FP32,n);
  GrB_Vector t2; GrB_Vector_new(&t2,GrB_FP32,n);
  GrB_Vector t3; GrB_Vector_new(&t3,GrB_FP32,n);
  GrB_Vector t4; GrB_Vector_new(&t4,GrB_FP32,n);
  for(int i=d-1; i>0; i--)
  {
    GrB_Vector_assign_FP32(t1,GrB_NULL,GrB_NULL,1.0f,GrB_ALL,n,GrB_NULL);          // t1 = 1+delta
    GrB_Vector_eWiseAdd_Monoid(t1,GrB_NULL,GrB_NULL,FP32Add,t1,*delta,GrB_NULL);
    GrB_Col_extract(t2,GrB_NULL,GrB_NULL,sigma,GrB_ALL,n,i,tr1);           // t2 = sigma[i,:]
    GrB_Vector_eWiseMult_BinaryOp(t2,GrB_NULL,GrB_NULL,GrB_DIV_FP32,t1,t2,GrB_NULL);   // t2 = (1+delta)/sigma[i,:]
    GrB_mxv(t3,GrB_NULL,GrB_NULL,FP32AddMul,A,t2,GrB_NULL);            // add contributions made by
                                                                       //     successors of a node
    GrB_Col_extract(t4,GrB_NULL,GrB_NULL,sigma,GrB_ALL,n,i-1,tr1);         // t4 = sigma[i-1,:]
    GrB_Vector_eWiseMult_Monoid(t4,GrB_NULL,GrB_NULL,FP32Mul,t4,t3,GrB_NULL);        // t4 = sigma[i-1,:]*t3
    GrB_Vector_eWiseAdd_Monoid(*delta,GrB_NULL,GrB_NULL,FP32Add,*delta,t4,GrB_NULL); // accumulate into delta
  }

  GrB_Matrix_free(&sigma);
  GrB_Vector_free(&q);
  GrB_Vector_free(&p);
  GrB_Semiring_free(&Int32AddMul);
  GrB_Monoid_free(&Int32Add);
  GrB_Semiring_free(&FP32AddMul);
  GrB_Monoid_free(&FP32Add);
  GrB_Monoid_free(&FP32Mul);
  GrB_Descriptor_free(&desc);
  GrB_Vector_free(&t1);
  GrB_Vector_free(&t2);
  GrB_Vector_free(&t3);
  GrB_Vector_free(&t4);

  return GrB_SUCCESS;
}

//****************************************************************************
// Compute partial BC metric for a subset of source vertices, s, in graph A
GrB_Info BC_GB_update(GrB_Vector *delta, GrB_Matrix A, GrB_Index *s, GrB_Index nsver)
{
  GrB_Index n;
  GrB_Matrix_nrows(&n, A);                             // n = # of vertices in graph
  GrB_Vector_new(delta,GrB_FP32,n);                    // Vector<float> delta(n)

  GrB_Monoid Int32Add;                                 // Monoid <int32_t,+,0>
  GrB_Monoid_new_INT32(&Int32Add,GrB_PLUS_INT32,0);
  GrB_Semiring Int32AddMul;                            // Semiring <int32_t,int32_t,int32_t,+,*,0>
  GrB_Semiring_new(&Int32AddMul,Int32Add,GrB_TIMES_INT32);

  // Descriptor for BFS phase mxm
  GrB_Descriptor desc_tsr;
  GrB_Descriptor_new(&desc_tsr);
  GrB_Descriptor_set(desc_tsr,GrB_INP0,GrB_TRAN);      // transpose the adjacency matrix
  GrB_Descriptor_set(desc_tsr,GrB_MASK,GrB_COMP);      // complement the mask
  GrB_Descriptor_set(desc_tsr,GrB_OUTP,GrB_REPLACE);   // clear output before result is stored

  // index and value arrays needed to build numsp
  GrB_Index *i_nsver = (GrB_Index*)malloc(sizeof(GrB_Index)*nsver);
  int32_t   *ones    = (int32_t*)  malloc(sizeof(int32_t)*nsver);
  for(int i=0; i<nsver; ++i) {
    i_nsver[i] = i;
    ones[i] = 1;
  }

  // numsp: structure holds the number of shortest paths for each node and starting vertex
  // discovered so far.  Initialized to source vertices:  numsp[s[i],i]=1, i=[0,nsver)
  GrB_Matrix numsp;
  GrB_Matrix_new(&numsp, GrB_INT32, n, nsver);
  GrB_Matrix_build_INT32(numsp,s,i_nsver,ones,nsver,GrB_PLUS_INT32);
  free(i_nsver); free(ones);

  // frontier: Holds the current frontier where values are path counts.
  // Initialized to out vertices of each source node in s.
  GrB_Matrix frontier;
  GrB_Matrix_new(&frontier, GrB_INT32, n, nsver);
  GrB_Matrix_extract(frontier,numsp,GrB_NULL,A,GrB_ALL,n,s,nsver,desc_tsr);

  // sigma: stores frontier information for each level of BFS phase.  The memory
  // for an entry in sigmas is only allocated within the do-while loop if needed
  GrB_Matrix *sigmas = (GrB_Matrix*)malloc(sizeof(GrB_Matrix)*n);   // n is an upper bound on diameter

  int32_t   d = 0;                                       // BFS level number
  GrB_Index nvals = 0;                                   // nvals == 0 when BFS phase is complete

  // --------------------- The BFS phase (forward sweep) ---------------------------
  do {
    // sigmas[d](:,s) = d^th level frontier from source vertex s
    GrB_Matrix_new(&(sigmas[d]), GrB_BOOL, n, nsver);

    GrB_Matrix_apply(sigmas[d],GrB_NULL,GrB_NULL,
              GrB_IDENTITY_BOOL,frontier,GrB_NULL);    // sigmas[d](:,:) = (Boolean) frontier
    GrB_Matrix_eWiseAdd_Monoid(numsp,GrB_NULL,GrB_NULL,
                 Int32Add,numsp,frontier,GrB_NULL);    // numsp += frontier (accum path counts)
    GrB_mxm(frontier,numsp,GrB_NULL,
            Int32AddMul,A,frontier,desc_tsr);          // f<!numsp> = A' +.* f (update frontier)
    GrB_Matrix_nvals(&nvals,frontier);                 // number of nodes in frontier at this level
    d++;
  } while (nvals);

  GrB_Monoid FP32Add;                                  // Monoid <float,+,0.0>
  GrB_Monoid_new_FP32(&FP32Add,GrB_PLUS_FP32,0.0f);
  GrB_Monoid FP32Mul;                                  // Monoid <float,*,1.0>
  GrB_Monoid_new_FP32(&FP32Mul,GrB_TIMES_FP32,1.0f);
  GrB_Semiring FP32AddMul;                             // Semiring <float,float,float,+,*,0.0>
  GrB_Semiring_new(&FP32AddMul,FP32Add,GrB_TIMES_FP32);

  // nspinv: the inverse of the number of shortest paths for each node and starting vertex.  |\label{line:nspinv}|
  GrB_Matrix nspinv;
  GrB_Matrix_new(&nspinv,GrB_FP32,n,nsver);
  GrB_Matrix_apply(nspinv,GrB_NULL,GrB_NULL,
            GrB_MINV_FP32,numsp,GrB_NULL);             // nspinv = 1./numsp

  // bcu: BC updates for each vertex for each starting vertex in s
  GrB_Matrix bcu;
  GrB_Matrix_new(&bcu,GrB_FP32,n,nsver);
  GrB_Matrix_assign_FP32(bcu,GrB_NULL,GrB_NULL,
             1.0f,GrB_ALL,n, GrB_ALL,nsver,GrB_NULL);  // filled with 1 to avoid sparsity issues

  // Descriptor used in the tally phase
  GrB_Descriptor desc_r;
  GrB_Descriptor_new(&desc_r);
  GrB_Descriptor_set(desc_r,GrB_OUTP,GrB_REPLACE);     // clear output before result is stored

  GrB_Matrix w;                                        // temporary workspace matrix
  GrB_Matrix_new(&w,GrB_FP32,n,nsver);

  // -------------------- Tally phase (backward sweep) --------------------
  for (int i=d-1; i>0; i--)  {
    GrB_Matrix_eWiseMult_Monoid(w,sigmas[i],GrB_NULL,
                  FP32Mul,bcu,nspinv,desc_r);          // w<sigmas[i]>=(1 ./ nsp).*bcu

    // add contributions by successors and mask with that BFS level's frontier
    GrB_mxm(w,sigmas[i-1],GrB_NULL,
            FP32AddMul,A,w,desc_r);                    // w<sigmas[i-1]> = (A +.* w)
    GrB_Matrix_eWiseMult_Monoid(bcu,GrB_NULL,GrB_PLUS_FP32,
                  FP32Mul,w,numsp,GrB_NULL);           // bcu += w .* numsp
  }

  // subtract "nsver" from every entry in delta (account for 1 extra value per bcu element)
  GrB_Vector_assign_FP32(*delta,GrB_NULL,GrB_NULL,
             -(float)nsver,GrB_ALL,n,GrB_NULL);        // fill with -nsver
  GrB_Matrix_reduce_BinaryOp(*delta,GrB_NULL,GrB_PLUS_FP32,
             GrB_PLUS_FP32,bcu,GrB_NULL);              // add all updates to -nsver

  // Release resources
  for(int i=0; i<d; i++) {
    GrB_Matrix_free(&(sigmas[i]));
  }
  free(sigmas);

  GrB_Matrix_free(&frontier);
  GrB_Matrix_free(&numsp);
  GrB_Matrix_free(&nspinv);
  GrB_Matrix_free(&bcu);
  GrB_Matrix_free(&w);
  GrB_Descriptor_free(&desc_tsr);
  GrB_Descriptor_free(&desc_r);
  GrB_Semiring_free(&Int32AddMul);
  GrB_Monoid_free(&Int32Add);
  GrB_Semiring_free(&FP32AddMul);
  GrB_Monoid_free(&FP32Add);
  GrB_Monoid_free(&FP32Mul);

  return GrB_SUCCESS;
}

#endif