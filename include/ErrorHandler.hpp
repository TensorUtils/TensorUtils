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

#ifndef ERRORHANDLER_HPP
#define ERRORHANDLER_HPP

#include <stdexcept>
#include <string>

namespace TensorUtils
{
    /*!
        \addtogroup TensorUtils
        @{
    */
    /*!
        \addtogroup ErrorHandler
        @{
        \brief This namespace contains error handler classes that inherit from "std::runtime_error".
        Most error handling is enabled only for the debug library "libtensor_utilsd.so".

        TensorUtils provides error handling to trace down rank or shape mismatches, invalid indices
        and invalid file paths.
        @code
        #include "TensorUtils.hpp"
        #include <iostream>

        using namespace std;
        using namespace TensorUtils;
        using namespace ErrorHandler;

        int main()
        {
            tensor<double> A;

            // READING FILES
            try
            {
                A.read("my_tensor.txt");
            }
            catch(UnableToOpenFile &ex) // unable to open file
            {
                cerr << ex.what() << endl;
            }
            catch(ShapeMismatch & ex)   // shape does not match data: corrupted file?
            {
                throw ex;
            }
            catch(exception & ex) // catch any other exception
            {
                throw ex;
            }

            // ACCESSING COMPONENTS
            A.alloc({2,3,5,7},1.0);
            try
            {
                A(1,2);         // OK! Returns A(1,2,0,0) by reference!
                A(0,0,0,0,0);   // too many indices: throws ShapeMismatch
                A(1,2,4,7);     // index out of range: throws std::out_of_range
            }
            catch(ShapeMismatch &ex) // more indices than expected!
            {
                cerr << ex.what() << endl;
            }
            catch(out_of_range &ex) //at least one index is out of range
            {
                cerr << ex.what() << endl;
            }

            // OPERATORS AND MEMBER FUNCTIONS
            tensor<double>          B({2,3,5,8},1.0);
            tensor<float>           C({2*3,5*7},1.0);
            tensor<long double>     D({},1.0); // scalar
            tensor<int,3>           E({3,5,7},1.0);
            tensor<unsigned long>   F({3,5,7},1.0);
            try
            {
                A += B; // different number of components: throws ShapeMismatch.
                A += C; // OK! Same number of elements, but different shapes!
                E = A;  // RankMismatch: unable to assign with a tensor of different rank!
                E = F;  // OK! Different types, but the ranks are the same.
                A = E;  // OK! A can have arbitrary rank.

                D = D[0];           // ShapeMismatch: don't try to slice scalars!
                E.alloc({2,3,5,7}); // RankMismatch: E has a fixed rank!

                A.alloc({2,3,5,7},1.0);
                A.assign(B, {1,2}, {1,2});  // ShapeMismatch: assignment with sub-tensor of invalid shape.
                A.assign(C, {1,2}, {0});    // OK! Same number of elements.
                A.assign(C, {1,3}, {0});    // invalid index: throws std::out_of_range.

                F = F.transpose({0,2,1}); // OK! Swap last two axes.
                F = F.transpose({1,3,2}); // ShapeMismatch: Reshape must be a permutation of (0,1,...,N-1).

                C = A.dot(A, {1,2,3}, {1,2,3,4});   // ShapeMismatch: axes must have the same size as the shapes.

                C = A.dot(A, {1,2,3,4}, {5,6,7,8}, {0,0,0,7});  // invalid index: std::out_of_range.
            }
            catch(ShapeMismatch &ex)
            {
                cerr << ex.what() << endl;
            }
            catch(RankMismatch &ex)
            {
                cerr << ex.what() << endl;
            }
            catch(out_of_range &ex)
            {
                cerr << ex.what() << endl;
            }

            return 0;
        }
        @endcode
    */
    namespace ErrorHandler
    {
        /*!
            \addtogroup ErrorHandler
            @{
        */

        //! This error is thrown, if a file cannot be opened. Inherits from std::runtime_error.
        /*!
            See \ref ErrorHandler for details.
        */
        class UnableToOpenFile : public std::runtime_error
        {
            public:
            //! Constructor inherited from std::runtime_error.
            explicit UnableToOpenFile (const std::string& what_arg) : std::runtime_error(what_arg) {};
        };

        //! This error is thrown if any tensor operation is called with invalid shapes or an invalid number of indices. Inherits from std::runtime_error.
        /*! If an index is out of range, std::out_of_range is thrown instead.
            Invalid usage of tensors with fixed ranks have their own error class \ref RankMismatch.
            See \ref ErrorHandler for details.
        */
        class ShapeMismatch : public std::runtime_error
        {
            public:
            //! Constructor inherited from std::runtime_error.
            explicit ShapeMismatch (const std::string& what_arg) : std::runtime_error(what_arg) {};
        };

        //! This error is thrown if any method would change the rank of a tensor with fixed rank. Inherits from std::runtime_error.
        /*!
            See \ref ErrorHandler for details.
        */
        class RankMismatch : public std::runtime_error
        {
            public:
            //! Constructor inherited from std::runtime_error.
            explicit RankMismatch (const std::string& what_arg) : std::runtime_error(what_arg) {};
        };
        /*! @} */
    }
    /*! @} */
    /*! @} */
}

#endif // ERRORHANDLER_HPP
