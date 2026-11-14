#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "linearLayer.h"

struct LinearLayerBundle {
    LinearLayer *main = nullptr;
    LinearLayer *reference = nullptr;
};

class LayerFactory {
  public:
    static LinearLayerBundle create(const std::string &layer_name,
                                    std::size_t input_size, std::size_t output_size,
                                    uint32_t *fallback_weight, std::size_t learner = 0);
};
