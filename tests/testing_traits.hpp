template <typename Op> struct ksize_t {
    template <typename P> auto operator()(const P &p) const
    {
        return Op::ksize(p.ksize_h, p.ksize_w);
    }
};

template <typename Op> struct stride_t {
    template <typename P> auto operator()(const P &p) const
    {
        return Op::stride(p.stride_h, p.stride_w);
    }
};

template <typename Op> struct padding_t {
    template <typename P> auto operator()(const P &p) const
    {
        return Op::padding(Op::padding_1d(p.pad_h_left, p.pad_h_right),
                           Op::padding_1d(p.pad_w_left, p.pad_w_right));
    }
};

template <typename Op> struct rate_t {
    template <typename P> auto operator()(const P &p) const
    {
        return Op::rate(p.rate_h, p.rate_w);
    }
};

template <typename Op, typename P> auto ksize(const P &p)
{
    return ksize_t<Op>()(p);
}

template <typename Op, typename P> auto stride(const P &p)
{
    return stride_t<Op>()(p);
}

template <typename Op, typename P> auto padding(const P &p)
{
    return padding_t<Op>()(p);
}

template <typename Op, typename P> auto rate(const P &p)
{
    return rate_t<Op>()(p);
}
