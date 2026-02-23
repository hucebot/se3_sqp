#include <trajopt/functions/abstract_function.h>
#include <trajopt/node.h>

MatrixXdConstRef AbstractFunction::get_jac_x() const {
    return _jacobian.leftCols(_node->ndx());
}

MatrixXdConstRef AbstractFunction::get_jac_u() const {
    return _jacobian.rightCols(_node->ndu());
}
