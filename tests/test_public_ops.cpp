#include <ttl/nn/ops>
#include <ttl/nn/testing>

template <typename F>
void test_op_exist()
{
    static_assert(sizeof(F) > 0, "");
}

#define TEST_OP_EXIST(op) test_op_exist<ttl::nn::ops::op>();

#define _ ,

TEST(test_public_ops, test_public_symbols)
{
    // elementary
    TEST_OP_EXIST(add);
    TEST_OP_EXIST(identity);
    TEST_OP_EXIST(mul);
    TEST_OP_EXIST(noop);
    TEST_OP_EXIST(sub);

    // init
    TEST_OP_EXIST(ones);
    TEST_OP_EXIST(uniform_constant);
    TEST_OP_EXIST(zeros);
    TEST_OP_EXIST(truncated_normal);
    // TEST_OP_EXIST(constant);

    // linear algebra
    // * level-1
    TEST_OP_EXIST(axpy);

    // * level-2
    // TEST_OP_EXIST(mv);
    // TEST_OP_EXIST(vm);

    // * level-3
    TEST_OP_EXIST(matmul);
    // TEST_OP_EXIST(mmt);
    // TEST_OP_EXIST(mtm);
    // TEST_OP_EXIST(mm);

    // io
    TEST_OP_EXIST(readfile);
    TEST_OP_EXIST(writefile);
    TEST_OP_EXIST(readtar);

    // nn
    // * conv
    TEST_OP_EXIST(conv<ttl::nn::ops::nhwc _ ttl::nn::ops::rscd>);
    TEST_OP_EXIST(conv<ttl::nn::ops::nchw _ ttl::nn::ops::dcrs>);
    TEST_OP_EXIST(conv1d);
    // * bias
    TEST_OP_EXIST(add_bias<ttl::nn::ops::hw>);
    TEST_OP_EXIST(add_bias<ttl::nn::ops::nhwc>);
    TEST_OP_EXIST(add_bias<ttl::nn::ops::nchw>);
    // * bn
    TEST_OP_EXIST(batch_norm<ttl::nn::ops::nhwc>);
    TEST_OP_EXIST(batch_norm<ttl::nn::ops::nchw>);
    // * pooling
    TEST_OP_EXIST(pool<ttl::nn::ops::pool_max _ ttl::nn::ops::nhwc>);
    TEST_OP_EXIST(pool<ttl::nn::ops::pool_mean _ ttl::nn::ops::nchw>);

    // * activation
    TEST_OP_EXIST(relu);
    TEST_OP_EXIST(softmax);
    // * loss
    TEST_OP_EXIST(xentropy);
    // * utility
    TEST_OP_EXIST(argmax);
    TEST_OP_EXIST(cast);
    TEST_OP_EXIST(onehot);
    TEST_OP_EXIST(similarity);
    TEST_OP_EXIST(top);
}

TEST(test_public_ops, test_full_names) {}
