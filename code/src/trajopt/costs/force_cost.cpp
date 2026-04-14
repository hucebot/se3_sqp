#include <trajopt/costs/force_cost.h>

ForceCost::ForceCost(const std::string& frame_name, const Vector3d& f_ref, double weight):
    AbstractCost(weight), _f_ref(f_ref), _contact_idx(-1)
{
    _name = frame_name + "_force_cost";
    _frame_name = frame_name;
}

ForceCost::ForceCost(const std::string& frame_name, const Vector3d& f_ref, const Matrix3d& weight):
AbstractCost(weight), _f_ref(f_ref), _contact_idx(-1)
{
    _name = frame_name + "_force_cost";
    _frame_name = frame_name;
}


void ForceCost::allocate_dims()
{
    // Find contact index in node's contact list
    _contact_idx = -1;
    const auto& contacts = _node->contacts();
    for (int i = 0; i < contacts.size(); ++i) {
        if (contacts[i].name == _frame_name) {
            _contact_idx = i;
            break;
        }
    }

    if (_contact_idx == -1) {
        throw std::runtime_error("ForceCost: frame '" + _frame_name +
                                 "' not registered as contact in node. Call node->add_contact() first.");
    }

    const int nv = _node->nv();

    _output_dim = 3;
    _input_dim = _node->ndx() + _node->ndu();

    if (_f_ref.size() == 0) {
        _f_ref.resize(_output_dim);
        _f_ref.setZero();
    }
}

void ForceCost::evaluate_impl()
{
    // Get force in local frame from control vector
    _f_local = _node->fc(_contact_idx);

    _value = _f_local - _f_ref;
}

void ForceCost::jacobian_impl()
{
    const int force_idx = _node->nv() + 3 * _contact_idx;
    _jacobian.setZero();
    _jacobian.middleCols(_node->ndx() + force_idx, 3).setIdentity();
}