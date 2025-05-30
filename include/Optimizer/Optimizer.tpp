#pragma once

template <typename Derived>
void Optimizer::_clipGradient(Eigen::MatrixBase<Derived>& gradient) const {
    if (_maxGradNorm <= 0.0) return;
    double norm = gradient.norm();
    if (norm == 0.0) return;
    if (norm > _maxGradNorm) {
        double scale = _maxGradNorm / std::max(norm, 1e-8);
        gradient *= scale;
    }
}

template <typename Derived>
void Optimizer::_applyWeightDecay(Eigen::MatrixBase<Derived>& grad, const Eigen::MatrixBase<Derived>& weights) const {
    if (_weightDecay > 0) {
        grad += _weightDecay * weights;
    }
}
