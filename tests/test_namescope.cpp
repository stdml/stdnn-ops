#include <nn/models>
#include <nn/testing>

TEST(namescope_sample_test, test_1)
{
    nn::models::namescope ns("::");
    std::string name;
    auto value = ns.with("model", [&] {
        return ns.with("l1", [&] {
            name = ns("b1");
            return *ns;
        });
    });
    ASSERT_EQ(*ns, "");
    ASSERT_EQ(ns("xxx"), "xxx");
    ASSERT_EQ(value, "model::l1");
    ASSERT_EQ(name, "model::l1::b1");
}
