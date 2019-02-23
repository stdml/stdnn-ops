#include <nn/ops-experimental.h>

int main()
{
    shape_t *s1 = 0;
    shape_t *s2 = 0;
    operator_t *op1 = 0;
    shape_t *shapes[] = {s1, s2};
    shape_t *s3 = op_infer(op1, 2, shapes);
    return 0;
}
