#include <trajopt/costs/step_cost.h>

StepCost::StepCost(const std::string& frame_name,
         double step_height_ref,
         double ground_ref,
         double weight):
    AbstractCost(weight), _ftc(frame_name, Eigen::Vector3d::Zero(), weight),
    _frame_name(frame_name), _step_height_ref(step_height_ref), _ground_ref(ground_ref)
{
    _name = "step_cost";
}

StepCost::StepCost(const std::string& frame_name,
         double step_height_ref,
         double ground_ref,
         const Matrix3d& weight):
    AbstractCost(weight), _ftc(frame_name, Eigen::Vector3d::Zero(), weight),
    _frame_name(frame_name), _step_height_ref(step_height_ref), _ground_ref(ground_ref)
{
    _name = "step_cost";
}


void StepCost::allocate_dims()
{
    _ftc.allocate_slices();

    _output_dim = 1;
    _input_dim = _ftc.get_input_dim();

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
        throw std::runtime_error("StepCost: frame '" + _frame_name +
                                 "' not registered as contact in node. Call node->add_contact() first.");
    }
}

void StepCost::evaluate_impl()
{
    _p_ref.setZero();

    // Update value based on contact state
    if (_node->contacts()[_contact_idx].active) {
        // Active contact: enforce friction cone
        _p_ref[2] = _ground_ref;
    } else {
        // Inactive contact: relax constraints
        _p_ref[2] = _step_height_ref;
    }

    _ftc.set_ref(_p_ref);
    _ftc.evaluate_impl();
    _value[0] = _ftc.get_value()[2];
}

void StepCost::jacobian_impl()
{
    _ftc.jacobian();
    _jacobian.leftCols(_node->ndx()) = _ftc.get_jac_x().row(2);
}
