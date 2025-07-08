CXX       := g++
CXXFLAGS  := -Wall -Wextra -std=c++17 -O2 -Iinclude -IC:/Libraries/Eigen-3.4.0

SRC := src\main.cpp \
src\MNISTLoader.cpp \
src\SimpleCNN.cpp \
src\Layers\Convolution2D.cpp \
src\Layers\FullyConnected.cpp \
src\Layers\MaxPooling.cpp \
src\LossFunction\LossFunction.cpp \
src\LossFunction\LossTypes.cpp \
src\Optimizer\Adam.cpp \
src\Optimizer\Optimizer.cpp \
src\Optimizer\SGD.cpp \
src\Regularization\BatchNormalization.cpp \
src\Regularization\Dropout.cpp
HEADERS_HPP := include/MNISTLoader.hpp \
include/SimpleCNN.hpp \
include/Activation/Activation.hpp \
include/Activation/ReLU.hpp \
include/Activation/Softmax.hpp \
include/Layers/Convolution2D.hpp \
include/Layers/FullyConnected.hpp \
include/Layers/MaxPooling.hpp \
include/LossFunction/LossFunction.hpp \
include/LossFunction/LossTypes.hpp \
include/Optimizer/Adam.hpp \
include/Optimizer/Optimizer.hpp \
include/Optimizer/SGD.hpp \
include/Regularization/BatchNormalization.hpp \
include/Regularization/Dropout.hpp
HEADERS_TPP := include\Activation\ReLU.tpp \
include\Activation\Softmax.tpp \
include\Layers\Convolution2D.tpp \
include\Layers\FullyConnected.tpp \
include\Optimizer\Optimizer.tpp \
include\Regularization\Dropout.tpp 
HEADERS := $(HEADERS_HPP) $(HEADERS_TPP)
OBJ     := $(SRC:.cpp=.o)
TARGET  := SimpleCNN.exe


all: $(TARGET)

$(TARGET): $(OBJ)
	@echo "[LD] $@"
	$(CXX) -o $@ $^

%.o: %.cpp
	@echo "[CC] $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	-@del /Q $(OBJ) $(TARGET) 2>nul || rm -f $(OBJ) $(TARGET)
 
format:
	clang-format -i $(SRC) $(HEADERS)

tidy:
	clang-tidy $(SRC) -- -Iinclude -IC:/Libraries/Eigen-3.4.0

.PHONY: all run clean format tidy
