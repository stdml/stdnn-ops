#include <vector>

#include <nn/ops-experimental.h>
#include <nn/ops>

struct shape_s {
};

struct tensor_ref_s {
};

struct tensor_view_s {
};

struct operator_s {
    using shape_list_t = std::vector<shape_t>;
    using tensor_view_list_t = std::vector<tensor_view_t>;

    virtual shape_t *operator()(const shape_list_t &shapes) const = 0;
    virtual void operator()(const tensor_ref_t &output,
                            const tensor_view_list_t &inputs) const = 0;
};

shape_t *op_infer(const operator_t *op, int arity, shape_t **shapes)
{
    operator_t::shape_list_t inputs;
    for (int i = 0; i < arity; ++i) { inputs.push_back(*(shapes[i])); }
    return (*op)(inputs);
}

void op_invoke(const operator_t *op, tensor_ref_t *output, int arity,
               tensor_view_t *tensors)
{
    operator_t::tensor_view_list_t inputs;
    for (int i = 0; i < arity; ++i) { inputs.push_back(tensors[i]); }
    (*op)(*output, inputs);
}
