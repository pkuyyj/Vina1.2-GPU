/*

   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Dr. Oleg Trott <ot14@columbia.edu>, 
           The Olson Lab, 
           The Scripps Research Institute

*/


#include "kernel.h"
#include "math.h"

/* Below based on mutate_cont.cpp */

void quaternion_increment(float* q, const float* rotation, float epsilon_fl);
void normalize_angle(float* x);

__device__ 
void output_type_cuda_init(output_type_cuda* out, __constant float* ptr) {
	for (int i = 0; i < 3; i++)out->position[i] = ptr[i];
	for (int i = 0; i < 4; i++)out->orientation[i] = ptr[i + 3];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)out->lig_torsion[i] = ptr[i + 3 + 4];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)out->flex_torsion[i] = ptr[i + 3 + 4 + MAX_NUM_OF_LIG_TORSION];
	out->lig_torsion_size = ptr[3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION];
	//did not assign coords and e
}

__device__ 
void output_type_cuda_init_with_output(output_type_cuda* out_new, const output_type_cuda* out_old) {
	for (int i = 0; i < 3; i++)out_new->position[i] = out_old->position[i];
	for (int i = 0; i < 4; i++)out_new->orientation[i] = out_old->orientation[i];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)out_new->lig_torsion[i] = out_old->lig_torsion[i];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)out_new->flex_torsion[i] = out_old->flex_torsion[i];
	out_new->lig_torsion_size = out_old->lig_torsion_size;
	//assign e but not coords
	out_new->e = out_old->e;
}

void output_type_cuda_increment(output_type_cuda* x, const change_cuda* c, float factor, float epsilon_fl) {
	// position increment
	for (int k = 0; k < 3; k++) x->position[k] += factor * c->position[k];
	// orientation increment
	float rotation[3];
	for (int k = 0; k < 3; k++) rotation[k] = factor * c->orientation[k];
	quaternion_increment(x->orientation, rotation, epsilon_fl);
	
	// torsion increment
	for (int k = 0; k < x->lig_torsion_size; k++) {
		float tmp = factor * c->lig_torsion[k];
		normalize_angle(&tmp);
		x->lig_torsion[k] += tmp;
		normalize_angle(&(x->lig_torsion[k]));
	}
}

__device__ 
float norm3(const float* a) {
	return sqrt(pown(a[0], 2) + pown(a[1], 2) + pown(a[2], 2));
}

__device__ 
void normalize_angle(float* x) {
	while (1) {
		if (*x >= -(M_PI) && *x <= (M_PI)) {
			break;
		}
		else if (*x > 3 * (M_PI)) {
			float n = (*x - (M_PI)) / (2 * (M_PI));
			*x -= 2 * (M_PI) * ceil(n);
		}
		else if (*x < 3 * -(M_PI)) {
			float n = (-*x - (M_PI)) / (2 * (M_PI));
			*x += 2 * (M_PI) * ceil(n);
		}
		else if (*x > (M_PI)) {
			*x -= 2 * (M_PI);
		}
		else if (*x < -(M_PI)) {
			*x += 2 * (M_PI);
		}
		else {
			break;
		}
	}
}

__device__ 
bool quaternion_is_normalized(float* q) {
	float q_pow = pown(q[0], 2) + pown(q[1], 2) + pown(q[2], 2) + pown(q[3], 2);
	float sqrt_q_pow = sqrt(q_pow);
	return (q_pow - 1 < 0.001) && (sqrt_q_pow - 1 < 0.001);
}

__device__ 
void angle_to_quaternion(float* q, const float* rotation, float epsilon_fl) {
	float angle = norm3(rotation);
	if (angle > epsilon_fl) {
		float axis[3] = { rotation[0] / angle, rotation[1] / angle ,rotation[2] / angle };
		if (norm3(axis) - 1 >= 0.001)printf("\nmutate: angle_to_quaternion() ERROR!"); // Replace assert(eq(axis.norm(), 1));
		normalize_angle(&angle);
		float c = cos(angle / 2);
		float s = sin(angle / 2);
		q[0] = c; q[1] = s * axis[0]; q[2] = s * axis[1]; q[3] = s * axis[2];
		return;
	}
	q[0] = 1; q[1] = 0; q[2] = 0; q[3] = 0;
	return;
}

// quaternion multiplication
__device__ 
void angle_to_quaternion_multi(float* qa, const float* qb) {
	float tmp[4] = { qa[0],qa[1],qa[2],qa[3] };
	qa[0] = tmp[0] * qb[0] - tmp[1] * qb[1] - tmp[2] * qb[2] - tmp[3] * qb[3];
	qa[1] = tmp[0] * qb[1] + tmp[1] * qb[0] + tmp[2] * qb[3] - tmp[3] * qb[2];
	qa[2] = tmp[0] * qb[2] - tmp[1] * qb[3] + tmp[2] * qb[0] + tmp[3] * qb[1];
	qa[3] = tmp[0] * qb[3] + tmp[1] * qb[2] - tmp[2] * qb[1] + tmp[3] * qb[0];
}

__device__ 
void quaternion_normalize_approx(float* q, float epsilon_fl) {
	const float s = pown(q[0], 2) + pown(q[1], 2) + pown(q[2], 2) + pown(q[3], 2);
	// Omit one assert()
	if (fabs(s - 1) < TOLERANCE)
		;
	else {
		const float a = sqrt(s);
		//if (a <= epsilon_fl) printf("\nmutate: quaternion_normalize_approx ERROR!"); // Replace assert(a > epsilon_fl);
		for (int i = 0; i < 4; i++)q[i] *= (1 / a);
		//if (quaternion_is_normalized(q) != true)printf("\nmutate: quaternion_normalize_approx() ERROR!");// Replace assert(quaternion_is_normalized(q));
	}
}

void quaternion_increment(float* q, const float* rotation, float epsilon_fl) {
	//if (quaternion_is_normalized(q) != true)printf("\nmutate: quaternion_increment() ERROR!"); // Replace assert(quaternion_is_normalized(q))
	float q_old[4] = { q[0],q[1],q[2],q[3] };
	angle_to_quaternion(q, rotation, epsilon_fl);
	angle_to_quaternion_multi(q, q_old);
	quaternion_normalize_approx(q, epsilon_fl);
}


__device__ 
float vec_distance_sqr(float* a, float* b) {
	return pown(a[0] - b[0], 2) + pown(a[1] - b[1], 2) + pown(a[2] - b[2], 2);
}

float gyration_radius(				int				m_lig_begin,
									int				m_lig_end,
						const		atom_cuda*		atoms,
						const		m_coords_cuda*	m_coords_gpu,
						const		float*			m_lig_node_origin
) {
	float acc = 0;
	int counter = 0;
	float origin[3] = { m_lig_node_origin[0], m_lig_node_origin[1], m_lig_node_origin[2] };
	for (int i = m_lig_begin; i < m_lig_end; i++) {
		float current_coords[3] = { m_coords_gpu->coords[i][0], m_coords_gpu->coords[i][1], m_coords_gpu->coords[i][2] };
		if (atoms[i].types[0] != EL_TYPE_H) { // for el, we use the first element (atoms[i].types[0])
			acc += vec_distance_sqr(current_coords, origin);
			++counter;
		}
	}
	return (counter > 0) ? sqrt(acc / counter) : 0;
}

void mutate_conf_cuda(const					int				step,
					const					int				num_steps,
											output_type_cuda*	c,
							__constant		int*			random_int_map_gpu,
							__constant		float			random_inside_sphere_map_gpu[][3],
							__constant		float*			random_fl_pi_map_gpu,
					const					int				m_lig_begin,
					const					int				m_lig_end,
					const					atom_cuda*		atoms,
					const					m_coords_cuda*	m_coords_gpu,
					const					float*			m_lig_node_origin_gpu,
					const					float			epsilon_fl,
					const					float			amplitude
) {

	int index = step; // global index (among all exhaus)
	int which = random_int_map_gpu[index];
	int lig_torsion_size = c->lig_torsion_size;
	int flex_torsion_size = 0; // FIX? 20210727
		if (which == 0) {
			for (int i = 0; i < 3; i++)
				c->position[i] += amplitude * random_inside_sphere_map_gpu[index][i];
			return;
		}
		--which;
		if (which == 0) {
			float gr = gyration_radius(m_lig_begin, m_lig_end, atoms, m_coords_gpu, m_lig_node_origin_gpu);
			if (gr > epsilon_fl) {
				float rotation[3];
				for (int i = 0; i < 3; i++)rotation[i] = amplitude / gr * random_inside_sphere_map_gpu[index][i];
				quaternion_increment(c->orientation, rotation, epsilon_fl);
			}
			return;
		}
		--which;
		if (which < lig_torsion_size) { c->lig_torsion[which] = random_fl_pi_map_gpu[index]; return; }
		which -= lig_torsion_size;

	if (flex_torsion_size != 0) {
		if (which < flex_torsion_size) { c->flex_torsion[which] = random_fl_pi_map_gpu[index]; return; }
		which -= flex_torsion_size;
	}
}

/*  Above based on mutate_conf.cpp */

/* Below based on matrix.cpp */

// symmetric matrix (only half of it are stored)
typedef struct {
	float data[MAX_HESSIAN_MATRIX_SIZE];
	int dim;
}matrix;

void matrix_init(matrix* m, int dim, float fill_data) {
	m->dim = dim;
	if ((dim * (dim + 1) / 2) > MAX_HESSIAN_MATRIX_SIZE)printf("\nnmatrix: matrix_init() ERROR!");
	((dim * (dim + 1) / 2)*sizeof(float)); // symmetric matrix
	for (int i = 0; i < (dim * (dim + 1) / 2); i++)m->data[i] = fill_data;
	for (int i = (dim * (dim + 1) / 2); i < MAX_HESSIAN_MATRIX_SIZE; i++)m->data[i] = 0;// Others will be 0
}

// as rugular 3x3 matrix
void mat_init(matrix* m, float fill_data) {
	m->dim = 3; // fixed to 3x3 matrix
	if (9 > MAX_HESSIAN_MATRIX_SIZE)printf("\nnmatrix: mat_init() ERROR!");
	for (int i = 0; i < 9; i++)m->data[i] = fill_data;
}


void matrix_set_diagonal(matrix* m, float fill_data) {
	for (int i = 0; i < m->dim; i++) {
		m->data[i + i * (i + 1) / 2] = fill_data;
	}
}

// as rugular matrix
__device__ 
void matrix_set_element(matrix* m, int dim, int x, int y, float fill_data) {
	m->data[x + y * dim] = fill_data;
}

__device__ 
void matrix_set_element_tri(matrix* m, int x, int y, float fill_data) {
	m->data[x + y*(y+1)/2] = fill_data;
}
__device__ 
int tri_index(int n, int i, int j) {
	if (j >= n || i > j)printf("\nmatrix: tri_index ERROR!");
	return i + j * (j + 1) / 2;
}

__device__ 
int index_permissive(const matrix* m, int i, int j) {
	return (i < j) ? tri_index(m->dim, i, j) : tri_index(m->dim, j, i);
}

/* Above based on matrix.cpp */

/* Below based on quasi_newton.cpp */

#define EL_TYPE_SIZE 11
#define AD_TYPE_SIZE 20
#define XS_TYPE_SIZE 17
#define SY_TYPE_SIZE 18



__device__ 
void change_cuda_init(change_cuda* g, const float* ptr) {
	for (int i = 0; i < 3; i++)g->position[i] = ptr[i];
	for (int i = 0; i < 3; i++)g->orientation[i] = ptr[i + 3];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)g->lig_torsion[i] = ptr[i + 3 + 3];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)g->flex_torsion[i] = ptr[i + 3 + 3 + MAX_NUM_OF_LIG_TORSION];
	g->lig_torsion_size = ptr[3 + 3 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION];
}

__device__ 
void change_cuda_init_with_change(change_cuda* g_new, const change_cuda* g_old) {
	for (int i = 0; i < 3; i++)g_new->position[i] = g_old->position[i];
	for (int i = 0; i < 3; i++)g_new->orientation[i] = g_old->orientation[i];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)g_new->lig_torsion[i] = g_old->lig_torsion[i];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)g_new->flex_torsion[i] = g_old->flex_torsion[i];
	g_new->lig_torsion_size = g_old->lig_torsion_size;
}

__device__ 
void output_type_cuda_init(output_type_cuda* out, __constant float* ptr); /* Function prototype in mutate_conf.cpp */
__device__ 
void output_type_cuda_init_with_output(output_type_cuda* out_new, const output_type_cuda* out_old); /* Function prototype in mutate_conf.cpp */

void print_ouput_type(output_type_cuda* x, int torsion_size) {
	for (int i = 0; i < 3; i++)printf("\nx.position[%d] = %0.16f", i, x->position[i]);
	for (int i = 0; i < 4; i++)printf("\nx.orientation[%d] = %0.16f", i, x->orientation[i]);
	for (int i = 0; i < torsion_size; i++)printf("\n x.torsion[%d] = %0.16f", i, x->lig_torsion[i]);
	printf("\n x.torsion_size = %f", x->lig_torsion_size);
}

void print_change(change_cuda* g, int torsion_size) {
	for (int i = 0; i < 3; i++)printf("\ng.position[%d] = %0.16f", i, g->position[i]);
	for (int i = 0; i < 3; i++)printf("\ng.orientation[%d] = %0.16f", i, g->orientation[i]);
	for (int i = 0; i < torsion_size; i++)printf("\ng.torsion[%d] = %0.16f", i, g->lig_torsion[i]);
	printf("\ng.torsion_size = %f", g->lig_torsion_size);
}

__device__ 
int num_atom_types(int atu) {
	switch (atu) {
	case 0: return EL_TYPE_SIZE;
	case 1: return AD_TYPE_SIZE;
	case 2: return XS_TYPE_SIZE;
	case 3: return SY_TYPE_SIZE;
	default: printf("Kernel1:num_atom_types() ERROR!"); return INFINITY;
	}
}

__device__ 
void elementwise_product(float* out, const float* a, const float* b) {
	out[0] = a[0] * b[0];
	out[1] = a[1] * b[1];
	out[2] = a[2] * b[2];
}

__device__ 
float elementwise_product_sum(const float* a, const float* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ 
float access_m_data(__constant float* m_data, int m_i, int m_j, int i, int j, int k) {
	return m_data[i + m_i * (j + m_j * k)];
}

__device__ 
bool not_max(float x) {
	return (x < 0.1 * INFINITY); /* Problem: replace max_fl with INFINITY? */
}

__device__ 
void curl_with_deriv(float* e, float* deriv, float v, const float epsilon_fl) {
	if (*e > 0 && not_max(v)) {
		float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		*e *= tmp;
		for (int i = 0; i < 3; i++)deriv[i] *= pown(tmp, 2);
	}
}

__device__ 
void curl_without_deriv(float* e, float v, const float epsilon_fl) {
	if (*e > 0 && not_max(v)) {
		float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		*e *= tmp;
	}
}

float g_evaluate(			__constant	grid_cuda*	g,
					const				float*		m_coords,				/* double[3] */
					const				float		slope,				/* double */
					const				float		v,					/* double */
										float*		deriv,				/* double[3] */
					const				float		epsilon_fl
) {
	int m_i = g->m_i;
	int m_j = g->m_j;
	int m_k = g->m_k;
	if(m_i * m_j * m_k == 0)printf("\nkernel2: g_evaluate ERROR!#1");
	float tmp_vec[3] = { m_coords[0] - g->m_init[0],m_coords[1] - g->m_init[1] ,m_coords[2] - g->m_init[2] };
	float tmp_vec2[3] = { g->m_factor[0],g->m_factor[1] ,g->m_factor[2] };
	float s[3];
	elementwise_product(s, tmp_vec, tmp_vec2); 

	float miss[3] = { 0,0,0 };
	int region[3];
	int a[3];
	int m_data_dims[3] = { m_i,m_j,m_k };
	for (int i = 0; i < 3; i++){
		if (s[i] < 0) {
			miss[i] = -s[i];
			region[i] = -1;
			a[i] = 0;
			s[i] = 0;
		}
		else if (s[i] >= g->m_dim_fl_minus_1[i]) {
			miss[i] = s[i] - g->m_dim_fl_minus_1[i];
			region[i] = 1;
			if (m_data_dims[i] < 2)printf("\nKernel2: g_evaluate ERROR!#2");
			a[i] = m_data_dims[i] - 2;
			s[i] = 1;
		}
		else {
			region[i] = 0;
			a[i] = (int)s[i];
			s[i] -= a[i];
		}
		if (s[i] < 0)
            printf("\nKernel2: g_evaluate ERROR!#3");
		if (s[i] > 1)
            printf("\nKernel2: g_evaluate ERROR!#4");
		if (a[i] < 0)
            printf("\nKernel2: g_evaluate ERROR!#5");
		if (a[i] + 1 >= m_data_dims[i])printf("\nKernel2: g_evaluate ERROR!#5");
	}

	float tmp_m_factor_inv[3] = { g->m_factor_inv[0],g->m_factor_inv[1],g->m_factor_inv[2] };
	const float penalty = slope * elementwise_product_sum(miss, tmp_m_factor_inv);
	if (penalty <= -epsilon_fl)printf("\nKernel2: g_evaluate ERROR!#6");

	const int x0 = a[0];
	const int y0 = a[1];
	const int z0 = a[2];
		 
	const int x1 = x0 + 1;
	const int y1 = y0 + 1;
	const int z1 = z0 + 1;

	const float f000 = access_m_data(g->m_data, m_i, m_j, x0, y0, z0);
	const float f100 = access_m_data(g->m_data, m_i, m_j, x1, y0, z0);
	const float f010 = access_m_data(g->m_data, m_i, m_j, x0, y1, z0);
	const float f110 = access_m_data(g->m_data, m_i, m_j, x1, y1, z0);
	const float f001 = access_m_data(g->m_data, m_i, m_j, x0, y0, z1);
	const float f101 = access_m_data(g->m_data, m_i, m_j, x1, y0, z1);
	const float f011 = access_m_data(g->m_data, m_i, m_j, x0, y1, z1);
	const float f111 = access_m_data(g->m_data, m_i, m_j, x1, y1, z1);


	const float x = s[0];
	const float y = s[1];
	const float z = s[2];
		  
	const float mx = 1 - x;
	const float my = 1 - y;
	const float mz = 1 - z;

	float f =
		f000 * mx * my * mz +
		f100 * x  * my * mz +
		f010 * mx * y  * mz +
		f110 * x  * y  * mz +
		f001 * mx * my * z	+
		f101 * x  * my * z	+
		f011 * mx * y  * z	+
		f111 * x  * y  * z  ;

	if (deriv) { 
		const float x_g =
			f000 * (-1) * my * mz +
			f100 *   1  * my * mz +
			f010 * (-1) * y  * mz +
			f110 *	 1  * y  * mz +
			f001 * (-1) * my * z  +
			f101 *   1  * my * z  +
			f011 * (-1) * y  * z  +
			f111 *   1  * y  * z  ;


		const float y_g =
			f000 * mx * (-1) * mz +
			f100 * x  * (-1) * mz +
			f010 * mx *   1  * mz +
			f110 * x  *   1  * mz +
			f001 * mx * (-1) * z  +
			f101 * x  * (-1) * z  +
			f011 * mx *   1  * z  +
			f111 * x  *   1  * z  ;


		const float z_g =
			f000 * mx * my * (-1) +
			f100 * x  * my * (-1) +
			f010 * mx * y  * (-1) +
			f110 * x  * y  * (-1) +
			f001 * mx * my *   1  +
			f101 * x  * my *   1  +
			f011 * mx * y  *   1  +
			f111 * x  * y  *   1  ;

		float gradient[3] = { x_g, y_g, z_g };

		curl_with_deriv(&f, gradient, v, epsilon_fl);

		float gradient_everywhere[3];

		for (int i = 0; i < 3; i++) {
			gradient_everywhere[i] = ((region[i] == 0) ? gradient[i] : 0);
			deriv[i] = g->m_factor[i] * gradient_everywhere[i] + slope * region[i];
		}
		return f + penalty;
	}	
	else {  /* none valid pointer */
		printf("\nKernel2: g_evaluate ERROR!#7");
		curl_without_deriv(&f, v, epsilon_fl);
		return f + penalty;
	}
}


float ig_eval_deriv(						output_type_cuda*		x,
											change_cuda*			g, 
						const				float				v,
								__constant	ig_cuda*				ig_cuda_gpu,
											m_cuda*				m_cuda_gpu,
						const				float				epsilon_fl
) {
	float e = 0;
	int nat = num_atom_types(ig_cuda_gpu->atu);
	for (int i = 0; i < m_cuda_gpu->m_num_movable_atoms; i++) {
		int t = m_cuda_gpu->atoms[i].types[ig_cuda_gpu->atu];
		if (t >= nat) {
			for (int j = 0; j < 3; j++)m_cuda_gpu->minus_forces.coords[i][j] = 0;
			continue;
		}
		float deriv[3];

		e = e + g_evaluate(&ig_cuda_gpu->grids[t], m_cuda_gpu->m_coords.coords[i], ig_cuda_gpu->slope, v, deriv, epsilon_fl);

		for (int j = 0; j < 3; j++) m_cuda_gpu->minus_forces.coords[i][j] = deriv[j];
	}
	return e;
}

__device__ 
void quaternion_to_r3(const float* q, float* orientation_m) {
	/* Omit assert(quaternion_is_normalized(q)); */
	const float a = q[0];
	const float b = q[1];
	const float c = q[2];
	const float d = q[3];

	const float aa = a * a;
	const float ab = a * b;
	const float ac = a * c;
	const float ad = a * d;
	const float bb = b * b;
	const float bc = b * c;
	const float bd = b * d;
	const float cc = c * c;
	const float cd = c * d;
	const float dd = d * d;

	/* Omit assert(eq(aa + bb + cc + dd, 1)); */
	matrix tmp;
	mat_init(&tmp, 0); /* matrix with fixed dimension 3(here we treate this as a regular matrix(not triangular matrix!)) */

	matrix_set_element(&tmp, 3, 0, 0,		(aa + bb - cc - dd)	);
	matrix_set_element(&tmp, 3, 0, 1, 2 *	(-ad + bc)			);
	matrix_set_element(&tmp, 3, 0, 2, 2 *	(ac + bd)			);
							 
	matrix_set_element(&tmp, 3, 1, 0, 2 *	(ad + bc)			);
	matrix_set_element(&tmp, 3, 1, 1,		(aa - bb + cc - dd)	);
	matrix_set_element(&tmp, 3, 1, 2, 2 *	(-ab + cd)			);
							 
	matrix_set_element(&tmp, 3, 2, 0, 2 *	(-ac + bd)			);
	matrix_set_element(&tmp, 3, 2, 1, 2 *	(ab + cd)			);
	matrix_set_element(&tmp, 3, 2, 2,		(aa - bb - cc + dd)	);

	for (int i = 0; i < 9; i++) orientation_m[i] = tmp.data[i];
}

__device__ 
void local_to_lab_direction(			float* out,
									const	float* local_direction,
									const	float* orientation_m
) {
	out[0] =	orientation_m[0] * local_direction[0] +
				orientation_m[3] * local_direction[1] +
				orientation_m[6] * local_direction[2];
	out[1] =	orientation_m[1] * local_direction[0] +
				orientation_m[4] * local_direction[1] +
				orientation_m[7] * local_direction[2];
	out[2] =	orientation_m[2] * local_direction[0] +
				orientation_m[5] * local_direction[1] +
				orientation_m[8] * local_direction[2];
}

__device__ 
void local_to_lab(						float*		out,
							const				float*		origin,
							const				float*		local_coords,
							const				float*		orientation_m
) {
	out[0] = origin[0] + (	orientation_m[0] * local_coords[0] +
							orientation_m[3] * local_coords[1] +
							orientation_m[6] * local_coords[2]
							);			 
	out[1] = origin[1] + (	orientation_m[1] * local_coords[0] +
							orientation_m[4] * local_coords[1] +
							orientation_m[7] * local_coords[2]
							);			 
	out[2] = origin[2] + (	orientation_m[2] * local_coords[0] +
							orientation_m[5] * local_coords[1] +
							orientation_m[8] * local_coords[2]
							);
}

__device__ 
void angle_to_quaternion2(				float*		out,
									const		float*		axis,
												float		angle
) {
	if (norm3(axis) - 1 >= 0.001)printf("\nkernel2: angle_to_quaternion() ERROR!"); /* Replace assert(eq(axis.norm(), 1)); */
	normalize_angle(&angle);
	float c = cos(angle / 2);
	float s = sin(angle / 2);
	out[0] = c;
	out[1] = s * axis[0];
	out[2] = s * axis[1];
	out[3] = s * axis[2];
}

void set(	const				output_type_cuda* x,
								rigid_cuda*		lig_rigid_gpu,
								m_coords_cuda*		m_coords_gpu,	
			const				atom_cuda*		atoms,				
			const				int				m_num_movable_atoms,
			const				float			epsilon_fl
) {
	
	for (int i = 0; i < 3; i++) lig_rigid_gpu->origin[0][i] = x->position[i];
	for (int i = 0; i < 4; i++) lig_rigid_gpu->orientation_q[0][i] = x->orientation[i]; 
	quaternion_to_r3(lig_rigid_gpu->orientation_q[0], lig_rigid_gpu->orientation_m[0]); /* set orientation_m */

	int begin = lig_rigid_gpu->atom_range[0][0];
	int end =	lig_rigid_gpu->atom_range[0][1];
	for (int i = begin; i < end; i++) {
		local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[0], &atoms[i].coords, lig_rigid_gpu->orientation_m[0]);
	}
	/* ************* end node.set_conf ************* */

	/* ************* branches_set_conf ************* */
	/* update nodes in depth-first order */
	for (int current = 1; current < lig_rigid_gpu->num_children + 1; current++) { /* current starts from 1 (namely starts from first child node) */
		int parent = lig_rigid_gpu->parent[current];
		float torsion = x->lig_torsion[current - 1]; /* torsions are all related to child nodes */
		local_to_lab(	lig_rigid_gpu->origin[current],
						lig_rigid_gpu->origin[parent],
						lig_rigid_gpu->relative_origin[current],
						lig_rigid_gpu->orientation_m[parent]
						); 
		local_to_lab_direction(	lig_rigid_gpu->axis[current],
								lig_rigid_gpu->relative_axis[current],
								lig_rigid_gpu->orientation_m[parent]
								); 
		float tmp[4];
		float parent_q[4] = {	lig_rigid_gpu->orientation_q[parent][0],
								lig_rigid_gpu->orientation_q[parent][1] ,
								lig_rigid_gpu->orientation_q[parent][2] ,
								lig_rigid_gpu->orientation_q[parent][3] };
		float current_axis[3] = {	lig_rigid_gpu->axis[current][0],
									lig_rigid_gpu->axis[current][1],
									lig_rigid_gpu->axis[current][2] };

		angle_to_quaternion2(tmp, current_axis, torsion);
		angle_to_quaternion_multi(tmp, parent_q);
		quaternion_normalize_approx(tmp, epsilon_fl);

		for (int i = 0; i < 4; i++) lig_rigid_gpu->orientation_q[current][i] = tmp[i]; /* set orientation_q */
		quaternion_to_r3(lig_rigid_gpu->orientation_q[current], lig_rigid_gpu->orientation_m[current]); /* set orientation_m */

		/* set coords */
		begin = lig_rigid_gpu->atom_range[current][0];
		end =	lig_rigid_gpu->atom_range[current][1];
		for (int i = begin; i < end; i++) {
			local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[current], &atoms[i].coords, lig_rigid_gpu->orientation_m[current]);
		}
	}
	/* ************* end branches_set_conf ************* */
}

void p_eval_deriv(						float*		out,
										int			type_pair_index,
										float		r2,
							__constant	p_cuda*		p_cuda_gpu,
					const				float		epsilon_fl
) {
	const float cutoff_sqr = p_cuda_gpu->m_cutoff_sqr;
	if(r2 > cutoff_sqr) printf("\nkernel2: p_eval_deriv() ERROR!");
	__constant p_m_data_cuda* tmp = &p_cuda_gpu->m_data[type_pair_index];
	float r2_factored = tmp->factor * r2;
	if (r2_factored + 1 >= SMOOTH_SIZE) printf("\nkernel2: p_eval_deriv() ERROR!");
	int i1 = (int)(r2_factored);
	int i2 = i1 + 1;
	if (i1 >= SMOOTH_SIZE || i1 < 0)printf("\n kernel2: p_eval_deriv() ERROR!");
	if (i2 >= SMOOTH_SIZE || i2 < 0)printf("\n : p_eval_deriv() ERROR!");
	float rem = r2_factored - i1;
	if (rem < -epsilon_fl)printf("\nkernel2: p_eval_deriv() ERROR!");
	if (rem >= 1 + epsilon_fl)printf("\nkernel2: p_eval_deriv() ERROR!");
	float p1[2] = { tmp->smooth[i1][0], tmp->smooth[i1][1] };
	float p2[2] = { tmp->smooth[i2][0], tmp->smooth[i2][1] };
	float e = p1[0] + rem * (p2[0] - p1[0]);
	float dor = p1[1] + rem * (p2[1] - p1[1]);
	out[0] = e;
	out[1] = dor;
}

__device__ 
void curl(float* e, float* deriv, float v, const float epsilon_fl) {
	if (*e > 0 && not_max(v)) {
		float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		(*e) = tmp * (*e);
		for (int i = 0; i < 3; i++)deriv[i] = deriv[i] * (tmp * tmp);
	}
}

float eval_interacting_pairs_deriv(		__constant	p_cuda*			p_cuda_gpu,
									const				float			v,
									const				lig_pairs_cuda*   pairs,
									const			 	m_coords_cuda*		m_coords,
														m_minus_forces* minus_forces,
									const				float			epsilon_fl
) {
	float e = 0;

	for (int i = 0; i < pairs->num_pairs; i++) {
		const int ip[3] = { pairs->type_pair_index[i], pairs->a[i] ,pairs->b[i] };
		float coords_b[3] = { m_coords->coords[ip[2]][0], m_coords->coords[ip[2]][1], m_coords->coords[ip[2]][2] };
		float coords_a[3] = { m_coords->coords[ip[1]][0], m_coords->coords[ip[1]][1], m_coords->coords[ip[1]][2] };
		float r[3] = { coords_b[0] - coords_a[0], coords_b[1] - coords_a[1] ,coords_b[2] - coords_a[2] };
		float r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
	
		if (r2 < p_cuda_gpu->m_cutoff_sqr) {
			float tmp[2];
			p_eval_deriv(tmp, ip[0], r2, p_cuda_gpu, epsilon_fl);
			float force[3] = { r[0] * tmp[1], r[1] * tmp[1] ,r[2] * tmp[1] };
			curl(&tmp[0], force, v, epsilon_fl);
			e += tmp[0];
			for (int j = 0; j < 3; j++)minus_forces->coords[ip[1]][j] -= force[j];
			for (int j = 0; j < 3; j++)minus_forces->coords[ip[2]][j] += force[j];
		}
	}
	return e;
}

__device__ 
void product(float* res, const float*a,const float*b) {
	res[0] = a[1] * b[2] - a[2] * b[1];
	res[1] = a[2] * b[0] - a[0] * b[2];
	res[2] = a[0] * b[1] - a[1] * b[0];
}

void POT_deriv(	const					m_minus_forces* minus_forces,
				const					rigid_cuda*		lig_rigid_gpu,
				const					m_coords_cuda*		m_coords,
										change_cuda*		g
) {
	int num_torsion = lig_rigid_gpu->num_children;
	int num_rigid = num_torsion + 1;
	float position_derivative_tmp[MAX_NUM_OF_RIGID][3];
	float position_derivative[MAX_NUM_OF_RIGID][3];
	float orientation_derivative_tmp[MAX_NUM_OF_RIGID][3];
	float orientation_derivative[MAX_NUM_OF_RIGID][3];
	float torsion_derivative[MAX_NUM_OF_RIGID]; /* torsion_derivative[0] has no meaning(root node has no torsion) */

	for (int i = 0; i < num_rigid; i++) {
		int begin = lig_rigid_gpu->atom_range[i][0];
		int end = lig_rigid_gpu->atom_range[i][1];
		for (int k = 0; k < 3; k++)position_derivative_tmp[i][k] = 0; 
		for (int k = 0; k < 3; k++)orientation_derivative_tmp[i][k] = 0;
		for (int j = begin; j < end; j++) {
			for (int k = 0; k < 3; k++)position_derivative_tmp[i][k] += minus_forces->coords[j][k];

			float tmp1[3] = {	m_coords->coords[j][0] - lig_rigid_gpu->origin[i][0],
								m_coords->coords[j][1] - lig_rigid_gpu->origin[i][1],
								m_coords->coords[j][2] - lig_rigid_gpu->origin[i][2] };
			float tmp2[3] = {  minus_forces->coords[j][0],
								minus_forces->coords[j][1],
								minus_forces->coords[j][2] };
			float tmp3[3];
			product(tmp3, tmp1, tmp2);
			for (int k = 0; k < 3; k++)
                orientation_derivative_tmp[i][k] += tmp3[k];
		}
	}

	/* position_derivative  */
	for (int i = num_rigid - 1; i >= 0; i--) { /* from bottom to top */
		for (int k = 0; k < 3; k++)position_derivative[i][k] = position_derivative_tmp[i][k]; 
		/* looking for chidren node */
		for (int j = 0; j < num_rigid; j++) {
			if (lig_rigid_gpu->children_map[i][j] == true) {
				for (int k = 0; k < 3; k++)position_derivative[i][k] += position_derivative[j][k]; /* self+children node */
			}
		}
	}

	/* orientation_derivetive */
	for (int i = num_rigid - 1; i >= 0; i--) { /* from bottom to top */
		for (int k = 0; k < 3; k++)orientation_derivative[i][k] = orientation_derivative_tmp[i][k];
		/* looking for chidren node */
		for (int j = 0; j < num_rigid; j++) {
			if (lig_rigid_gpu->children_map[i][j] == true) { /* self + children node + product */
				for (int k = 0; k < 3; k++)orientation_derivative[i][k] += orientation_derivative[j][k];
				float product_out[3];
				float origin_temp[3] = {	lig_rigid_gpu->origin[j][0] - lig_rigid_gpu->origin[i][0],
											lig_rigid_gpu->origin[j][1] - lig_rigid_gpu->origin[i][1],
											lig_rigid_gpu->origin[j][2] - lig_rigid_gpu->origin[i][2] };
				product(product_out, origin_temp, position_derivative[j]);
				for (int k = 0; k < 3; k++)orientation_derivative[i][k] += product_out[k];
			}
		}
	}

	/* torsion_derivative */
	for (int i = num_rigid - 1; i >= 0; i--) { 
		float sum = 0;
		for (int j = 0; j < 3; j++) sum += orientation_derivative[i][j] * lig_rigid_gpu->axis[i][j];
		torsion_derivative[i] = sum;
	}

	for (int k = 0; k < 3; k++)	g->position[k] = position_derivative[0][k];
	for (int k = 0; k < 3; k++) g->orientation[k] = orientation_derivative[0][k];
	for (int k = 0; k < num_torsion; k++) g->lig_torsion[k] = torsion_derivative[k + 1];
}


float m_eval_deriv(					output_type_cuda*		c,
										change_cuda*			g,
										m_cuda*				m_cuda_gpu,
							__constant	p_cuda*				p_cuda_gpu,
							__constant	ig_cuda*				ig_cuda_gpu,
					const	float*				v,
					const				float				epsilon_fl
) {
	set(c, &m_cuda_gpu->ligand.rigid, m_cuda_gpu->m_coords.coords, m_cuda_gpu->atoms, m_cuda_gpu->m_num_movable_atoms, epsilon_fl);

	float e = ig_eval_deriv(	c,
								g, 
								v[1],				
								ig_cuda_gpu,
								m_cuda_gpu,
								epsilon_fl							
							);
	
	e += eval_interacting_pairs_deriv(	p_cuda_gpu,
										v[0],
										&m_cuda_gpu->ligand.pairs,
										&m_cuda_gpu->m_coords,
										&m_cuda_gpu->minus_forces,
										epsilon_fl
									);

	POT_deriv(&m_cuda_gpu->minus_forces, &m_cuda_gpu->ligand.rigid, &m_cuda_gpu->m_coords, g);

	return e;
}


__device__ 
float find_change_index_read(const change_cuda* g, int index) {
	if (index < 3)return g->position[index];
	index -= 3;
	if (index < 3)return g->orientation[index];
	index -= 3;
	if (index < g->lig_torsion_size)return g->lig_torsion[index];
	printf("\nKernel2:find_change_index_read() ERROR!"); /* Shouldn't be here */
}

__device__ 
void find_change_index_write(change_cuda* g, int index, float data) {
	if (index < 3) { g->position[index] = data; return; }
	index -= 3;
	if (index < 3) { g->orientation[index] = data; return; }
	index -= 3;
	if (index < g->lig_torsion_size) { g->lig_torsion[index] = data; return; }
	printf("\nKernel2:find_change_index_write() ERROR!"); /* Shouldn't be here */
}

void minus_mat_vec_product(	const		matrix*		h,
							const		change_cuda*	in,
										change_cuda*  out
) {
	int n = h->dim;
	for (int i = 0; i < n; i++) {
		float sum = 0;
		for (int j = 0; j < n; j++) {
			sum += h->data[index_permissive(h, i, j)] * find_change_index_read(in, j);
		}
		find_change_index_write(out, i, -sum);
	}
}


__device__ 
float scalar_product(	const	change_cuda*			a,
								const	change_cuda*			b,
								int							n
) {
	float tmp = 0;
	for (int i = 0; i < n; i++) {
		tmp += find_change_index_read(a, i) * find_change_index_read(b, i);
	}
	return tmp;
}


float line_search(					 	m_cuda*				m_cuda_gpu,
							__constant	p_cuda*				p_cuda_gpu,
							__constant	ig_cuda*				ig_cuda_gpu,
										int					n,
					const				output_type_cuda*		x,
					const				change_cuda*			g,
					const				float				f0,
					const				change_cuda*			p,
										output_type_cuda*		x_new,
										change_cuda*			g_new,
										float*				f1,
					const				float				epsilon_fl,
					const	float*				hunt_cap
) {
	const float c0 = 0.0001;
	const int max_trials = 10;
	const float multiplier = 0.5;
	float alpha = 1;

	const float pg = scalar_product(p, g, n);

	for (int trial = 0; trial < max_trials; trial++) {

		output_type_cuda_init_with_output(x_new, x);

		output_type_cuda_increment(x_new, p, alpha, epsilon_fl);

		*f1 =  m_eval_deriv(x_new,
							g_new,
							m_cuda_gpu,
							p_cuda_gpu,
							ig_cuda_gpu,
							hunt_cap,
							epsilon_fl
							);

		if (*f1 - f0 < c0 * alpha * pg)
			break;
		alpha *= multiplier;
	}
	return alpha;
}


bool bfgs_update(			matrix*			h,
					const	change_cuda*		p,
					const	change_cuda*		y,
					const	float			alpha,
					const	float			epsilon_fl
) {

	const float yp = scalar_product(y, p, h->dim);
	
	if (alpha * yp < epsilon_fl) return false;
	change_cuda minus_hy;
	change_cuda_init_with_change(&minus_hy, y);
	minus_mat_vec_product(h, y, &minus_hy);
	const float yhy = -scalar_product(y, &minus_hy, h->dim);
	const float r = 1 / (alpha * yp);
	const int n = 6 + p->lig_torsion_size;

	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			float tmp = alpha * r * (find_change_index_read(&minus_hy, i) * find_change_index_read(p, j)
									+ find_change_index_read(&minus_hy, j) * find_change_index_read(p, i)) +
									+alpha * alpha * (r * r * yhy + r) * find_change_index_read(p, i) * find_change_index_read(p, j);

			h->data[i + j * (j + 1) / 2] += tmp;
		}
	}

	return true;
}



void bfgs(					output_type_cuda*			x,
								change_cuda*			g,
								m_cuda*				m_cuda_gpu,
					__constant	p_cuda*				p_cuda_gpu,
					__constant	ig_cuda*				ig_cuda_gpu,
			const	float*				hunt_cap,
			const				float				epsilon_fl,
			const				int					max_steps
) 
{
	int n = 3 + 3 + x->lig_torsion_size; /* the dimensions of matirx */

	matrix h;
	matrix_init(&h, n, 0);
	matrix_set_diagonal(&h, 1);

	change_cuda g_new;
	change_cuda_init_with_change(&g_new, g);

	output_type_cuda x_new;
	output_type_cuda_init_with_output(&x_new, x);
	 
	float f0 = m_eval_deriv(	x,
								g,
								m_cuda_gpu,
								p_cuda_gpu,
								ig_cuda_gpu,
								hunt_cap,
								epsilon_fl
							);

	float f_orig = f0;
	/* Init g_orig, x_orig */
	change_cuda g_orig;
	change_cuda_init_with_change(&g_orig, g);
	output_type_cuda x_orig;
	output_type_cuda_init_with_output(&x_orig, x);
	/* Init p */
	change_cuda p;
	change_cuda_init_with_change(&p, g);

	float f_values[MAX_NUM_OF_BFGS_STEPS + 1];
	f_values[0] = f0;

	for (int step = 0; step < max_steps; step++) {

		minus_mat_vec_product(&h, g, &p);
		float f1 = 0;

		const float alpha = line_search(	m_cuda_gpu,
											p_cuda_gpu,
											ig_cuda_gpu,
											n,
											x,
											g,
											f0,
											&p,
											&x_new,
											&g_new,
											&f1,
											epsilon_fl,
											hunt_cap
										);

		change_cuda y;
		change_cuda_init_with_change(&y, &g_new);
		/* subtract_change */
		for (int i = 0; i < n; i++) {
			float tmp = find_change_index_read(&y, i) - find_change_index_read(g, i);
			find_change_index_write(&y, i, tmp);
		}
		f_values[step + 1] = f1;
		f0 = f1;
		output_type_cuda_init_with_output(x, &x_new);
		if (!(sqrt(scalar_product(g, g, n)) >= 1e-5))break;
		change_cuda_init_with_change(g, &g_new);

		if (step == 0) {
			float yy = scalar_product(&y, &y, n);
			if (fabs(yy) > epsilon_fl) {
				matrix_set_diagonal(&h, alpha * scalar_product(&y, &p, n) / yy);
			}
		}

		bool h_updated = bfgs_update(&h, &p, &y, alpha, epsilon_fl);
	}

	if (!(f0 <= f_orig)) {
		f0 = f_orig;
		output_type_cuda_init_with_output(x, &x_orig);
		change_cuda_init_with_change(g, &g_orig);
	}

	// write output_type_cuda energy
	x->e = f0;
}


/* Above based on quasi_newton.cpp */

/* Below based on monte_carlo.cpp */

// put back results to vina
std::vector<output_type> monte_carlo::cuda_to_vina(output_type_cuda result_ptr[], int exhaus) const {
	std::vector<output_type> results_vina;
	for (int i = 0; i < exhaus; i++) {
		output_type_cuda tmp_cuda = result_ptr[i];
		conf tmp_c;
		tmp_c.ligands.resize(1);
		// Position
		for (int j = 0; j < 3; j++)
			tmp_c.ligands[0].rigid.position[j] = tmp_cuda.position[j];
		// Orientation
		qt q(tmp_cuda.orientation[0], tmp_cuda.orientation[1], tmp_cuda.orientation[2], tmp_cuda.orientation[3]);
		tmp_c.ligands[0].rigid.orientation = q;
		output_type tmp_vina(tmp_c, tmp_cuda.e);
		// torsion
		for (int j = 0; j < tmp_cuda.lig_torsion_size; j++)tmp_vina.c.ligands[0].torsions.push_back(tmp_cuda.lig_torsion[j]);
		// coords
		for (int j = 0; j < MAX_NUM_OF_ATOMS; j++) {
			vec v_tmp(tmp_cuda.coords[j][0], tmp_cuda.coords[j][1], tmp_cuda.coords[j][2]);
			if (v_tmp[0] * v_tmp[1] * v_tmp[2] != 0) tmp_vina.coords.push_back(v_tmp);
		}
		results_vina.push_back(tmp_vina);
	}
	return results_vina;
}

output_type monte_carlo::operator()(model& m, const precalculate_byatom& p, const igrid& ig, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const {
	output_container tmp;
	this->operator()(m, tmp, p, ig, corner1, corner2, increment_me, generator); // call the version that produces the whole container
	VINA_CHECK(!tmp.empty());
	return tmp.front();
}

bool metropolis_accept(fl old_f, fl new_f, fl temperature, rng& generator) {
	if(new_f < old_f) return true;
	const fl acceptance_probability = std::exp((old_f - new_f) / temperature);
	return random_fl(0, 1, generator) < acceptance_probability;
}

// out is sorted
void monte_carlo::operator()(model& m, output_container& out, const precalculate_byatom& p, const igrid& ig, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const {
    int evalcount = 0;
	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl
	conf_size s = m.get_size();
	change g(s);
	output_type tmp(s, 0);
	tmp.c.randomize(corner1, corner2, generator);
	fl best_e = max_fl;
	quasi_newton quasi_newton_par;
    quasi_newton_par.max_steps = local_steps;
	VINA_U_FOR(step, global_steps) {
		if(increment_me)
			++(*increment_me);
		if((max_evals > 0) & (evalcount > max_evals))
			break;
		output_type candidate = tmp;
		mutate_conf(candidate.c, m, mutation_amplitude, generator);
		quasi_newton_par(m, p, ig, candidate, g, hunt_cap, evalcount);
		if(step == 0 || metropolis_accept(tmp.e, candidate.e, temperature, generator)) {
			tmp = candidate;

			m.set(tmp.c); // FIXME? useless?

			// FIXME only for very promising ones
			if(tmp.e < best_e || out.size() < num_saved_mins) {
				quasi_newton_par(m, p, ig, tmp, g, authentic_v, evalcount);
				m.set(tmp.c); // FIXME? useless?
				tmp.coords = m.get_heavy_atom_movable_coords();
				add_to_output_container(out, tmp, min_rmsd, num_saved_mins); // 20 - max size
				if(tmp.e < best_e)
					best_e = tmp.e;
			}
		}
	}
	VINA_CHECK(!out.empty());
	VINA_CHECK(out.front().e <= out.back().e); // make sure the sorting worked in the correct order
}
