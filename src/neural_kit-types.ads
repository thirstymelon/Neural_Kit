package Neural_Kit.Types is
   pragma Pure (Neural_Kit.Types);

   subtype Scalar is Float;
   subtype Weight is Scalar;
   subtype Bias is Scalar;

   type Vector is array (Positive range <>) of Scalar;

   type Matrix is array (Positive range <>, Positive range <>) of Scalar;

   subtype Features_T is Vector;
   subtype Weights_T is Vector;
   subtype Scalar_T is Scalar;
   subtype Weight_T is Weight;
   subtype Bias_T is Bias;

   Dimension_Error : exception;
   Numerical_Error : exception;

end Neural_Kit.Types;
