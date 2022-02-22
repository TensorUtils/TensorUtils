/*
TensorUtils Version 0.1

Copyright 2022 Christoph Widder

This file is part of TensorUtils.

TensorUtils is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

TensorUtils is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with TensorUtils.
If not, see <https://www.gnu.org/licenses/>.

*/

#ifndef TENSORUTILS_HPP
#define TENSORUTILS_HPP

#include "ErrorHandler.hpp"
#include "TensorDerived.hpp"

/*! \mainpage TensorUtils Version 0.1
    \date 22.02.2022
    \author    Christoph Widder
    \copyright GNU Public License.


    TensorUtils is free software. See \ref License for the terms of use.
    You are welcome to report any bugs to <tensorutils@gmail.com>.

    \section intro Introduction

    TensorUtils presents a tensor class which is derived from std::vector<T>.
    It allows the usage of all std::vector routines, but has its own constructors.
    The tensor class allows to allocate, initialize, read and write tensors of floating or integral types up to rank 8.
    It provides text and binary file formats as well as element-wise operations with support for
    type conversions and chaining. The usage of this library might help to avoid memory leaks, segmentation faults,
    nested loops as well as error-prone index conversions. All methods are explicitly instantiated and stored in a shared library,
    which minimizes the compile time of your source code.

    Supported types for the components are the follogwing:

    DATA TYPE           | EXTENSION
    --------------------|-----------
    float               | .f32
    double              | .f64
    long double         | .f80
    unsigned char       | .uc
    signed char         | .sc
    unsigned short      | .us
    unsigned int        | .u
    unsigned long       | .ul
    unsigned long long  | .ull
    short               | .s
    int                 | .int
    long                | .l
    long long           | .ll

    The whole project is wrapped into the namespace \ref TensorUtils.
    See the main class \ref TensorUtils::TensorBase<T> for routines and examples.
    Although this base class is fully functional, it is recommended to use
    the derived class \ref TensorUtils::TensorDerived<T,N> which allows you to use
    tensors of arbitrary rank as well as tensors with fixed rank.
    This will be helpful if you need distinct types for tensors of different ranks.
    More details on error-handling can be found in \ref ErrorHandler.

    \section compile Compile

    From within the project folder, type:

        make

    This will create a shared library at:

        PATH_TO_TENSOR_UTILS/lib/Release/libtensor_utils.so
        PATH_TO_TENSOR_UTILS/lib/Debug/libtensor_utilsd.so

    \section install Installation (UBUNTU)

    If you don't want to install the library
    or if you don't want to use the default location, see \ref no_install.

    To install the library at the default locations "/usr/local/lib" and "/usr/local/include", type:

        sudo make install
        make clean

    The header files are now installed as read only (444) in:

        /usr/local/lib/tensor_utils

    The shaed library is installed with read and execute permissions (555) at:

        /usr/local/lib/libtensor_utils.so	# use this library for your release
        /usr/local/lib/libtensor_utilsd.so	# use this library for debugging

    To deinstall the library type:

        sudo make uninstall

    Include the header files:

        -I/usr/local/include/tensor_utils

    Link the shared library:

        -L/usr/local/lib/
        -ltensor_utils
        -ltensor_utilsd

    Your compile commands could look something like:

        # debug
        g++ -Wall -std=c++17 -fexceptions -g -I/usr/local/include/tensor_utils -c main.cpp -o obj/Debug/main.o
        g++ -L/usr/local/lib -o bin/Debug/main obj/Debug/main.o   -ltensor_utilsd

        # release
        g++ -Wall -std=c++17 -fexceptions -O3 -I/usr/local/include/tensor_utils -c main.cpp -o obj/Release/main.o
        g++ -L/usr/local/lib -o bin/Release/main obj/Release/main.o   -ltensor_utils

    You are ready to run your executable!


    \section no_install Usage without installation / Installation with user-defined paths

    Include the header files:

        -I/PATH_TO_TENSOR_UTILS/include

    Link the shared library:

        -L/PATH_TO_TENSOR_UTILS/lib/Release
        -L/PATH_TO_TENSOR_UTILS/lib/Debug
        -ltensor_utils
        -ltensor_utilsd

    Your compile commands could look something like:

        # debug
        g++ -Wall -std=c++17 -fexceptions -g -I/PATH_TO_TENSOR_UTILS/include -c main.cpp -o obj/Debug/main.o
        g++ -L/PATH_TO_TENSOR_UTILS/lib/Debug -o bin/Debug/main obj/Debug/main.o   -ltensor_utilsd

        # release
        g++ -Wall -std=c++17 -fexceptions -O3 -I/usr/local/include/tensor_utils -c main.cpp -o obj/Release/main.o
        g++ -L/PATH_TO_TENSOR_UTILS/lib/Release -o bin/Release/main obj/Release/main.o   -ltensor_utils

    To run your executable, you need to make sure that your operating system will find the shared library.

    On UBUNTU:

        # Release
        cd PATH_TO_TENSOR_UTILS/lib/Release
        export LD_LIBRARY_PATH="$(pwd)"

        # Debug
        cd PATH_TO_TENSOR_UTILS/lib/Debug
        export LD_LIBRARY_PATH="$(pwd)"

    You are ready to run your executable!

    In order to install the library path permanently, create a .conf file in

        /etc/ld.so.conf.d/your_config.conf

    add the following paths in this file

        PATH_TO_TENSOR_UTILS/lib/Release
        PATH_TO_TENSOR_UTILS/lib/Debug

    and update the cache:

        sudo ldconfig

    \section License

        TensorUtils Version 0.1

        Copyright 2022 Christoph Widder

        This file is part of TensorUtils.

        TensorUtils is free software: you can redistribute it and/or modify it under the terms of
        the GNU General Public License as published by the Free Software Foundation, either
        version 3 of the License, or (at your option) any later version.

        TensorUtils is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
        without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
        PURPOSE. See the GNU General Public License for more details.

        You should have received a copy of the GNU General Public License along with TensorUtils.
        If not, see <https://www.gnu.org/licenses/>.


*/

/*!
    \addtogroup TensorUtils
    @{ \brief This is the main namespace that wraps the entire implementation of this project.
*/
namespace TensorUtils
{
    /*!
        \addtogroup TensorUtils
        @{
    */

    /*!
        \brief Alias declaration for derived class "TensorDerived<T,N>",
        where "T" is the type of the components and "N" is the rank.
        "TensorDerived<T,N>" inherits all its functionality from the base
        class "TensorBase<T>".

        Construct tensors with arbitrary or fixed rank:
        \code
        #include "TensorUtils.hpp"

        int main()
        {
            TensorUtils::tensor<double> my_tensor;


            return 0;
        }
        \endcode
    */
    template<class T, int N=-1> using tensor = TensorDerived<T,N>;
    /*! @} */
}
/*! @} */

#endif // TENSORUTILS_HPP
