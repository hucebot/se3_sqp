#pragma once

#include <constraints/abstract_constraint.h>
#include <costs/abstract_cost.h>

#include <vector>

class Stage {
   private:
    std::vector<AbstractCost> cost_list;
    std::vector<AbstractConstraint> constraint_list;

   public:
    Stage(/* args */);
    ~Stage();
};

Stage::Stage(/* args */) {}

Stage::~Stage() {}
