#pragma once

#include <trajopt/functions/abstract_function.h>

/**
 * AbstractCost extends AbstractFunction with cost-specific semantics.
 *
 * Derived classes must implement:
 * - allocate_dims(): Set _output_dim, _input_dim, and custom members
 * - evaluate_impl(): Compute cost value
 * - jacobian_impl(): Compute first derivatives
 */
class AbstractCost : public AbstractFunction {
   protected:
    MatrixXd _weight;
    double _scalar_weight;
    MatrixXd _matrix_weight;

    void post_allocate() override {
        if (_matrix_weight.size() > 0) {
            if (_matrix_weight.rows() != _output_dim || _matrix_weight.cols() != _output_dim) {
                throw std::invalid_argument(
                    _name + ": Weight matrix must be " +
                    std::to_string(_output_dim) + "x" + std::to_string(_output_dim) +
                    ", but got " + std::to_string(_matrix_weight.rows()) + "x" +
                    std::to_string(_matrix_weight.cols()));
            }
            _weight = _matrix_weight;
        } else {
            _weight = MatrixXd::Identity(_output_dim, _output_dim) * _scalar_weight;
        }
    }

   public:
    AbstractCost(double weight = 1.0)
        : AbstractFunction(), _scalar_weight(weight) {}

    AbstractCost(const MatrixXd& weight)
        : AbstractFunction(), _scalar_weight(1.0), _matrix_weight(weight) {}

    virtual ~AbstractCost() {}

    const MatrixXd& get_weight() const { return _weight; }  // Return by ref - no copy!

    void set_weight(double w) {
        _scalar_weight = w;
        _weight = MatrixXd::Identity(_output_dim, _output_dim) * w;
    }

    void set_weight(const MatrixXd& w) {
        if (w.rows() != _output_dim || w.cols() != _output_dim) {
            throw std::invalid_argument(
                _name + ": Weight matrix must be " +
                std::to_string(_output_dim) + "x" + std::to_string(_output_dim) +
                ", but got " + std::to_string(w.rows()) + "x" + std::to_string(w.cols()));
        }
        _weight = w;
    }
};