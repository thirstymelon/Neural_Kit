package Neural_Kit.Types is

   subtype Scalar_T is Float;
   subtype Weight_T is Scalar_T;
   subtype Bias_T is Scalar_T;

   Feature_Count : constant Positive := 2;

   type Features_T is array (1 .. Feature_Count) of Scalar_T;
   type Weights_T is array (1 .. Feature_Count) of Weight_T;

end Neural_Kit.Types;
