#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct shape_s shape_t;
typedef struct tensor_ref_s tensor_ref_t;
typedef struct tensor_view_s tensor_view_t;
typedef struct operator_s operator_t;

// TODO: add dtype info to infer
// FIXME: assuming dtype is the same
extern shape_t *op_infer(const operator_t *op, int arity, shape_t **shapes);
extern void op_invoke(const operator_t *op, tensor_ref_t *output, int arity,
                      tensor_view_t *tensor);

#ifdef __cplusplus
}
#endif
