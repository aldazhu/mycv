/***********************************************************
 * File generated by the HALCON-Compiler hcomp version 20.11
 * Usage: Interface to C++
 *
 * Software by: MVTec Software GmbH, www.mvtec.com
 ***********************************************************/


#ifndef HCPP_HVARIATIONMODEL
#define HCPP_HVARIATIONMODEL

namespace HalconCpp
{

// Represents an instance of a variation model.
class LIntExport HVariationModel : public HHandle
{

public:

  // Create an uninitialized instance
  HVariationModel():HHandle() {}

  // Copy constructor
  HVariationModel(const HVariationModel& source) : HHandle(source) {}

  // Copy constructor
  HVariationModel(const HHandle& handle);

  // Create HVariationModel from handle, taking ownership
  explicit HVariationModel(Hlong handle);

  bool operator==(const HHandle& obj) const
  {
    return HHandleBase::operator==(obj);
  }

  bool operator!=(const HHandle& obj) const
  {
    return HHandleBase::operator!=(obj);
  }

protected:

  // Verify matching semantic type ('variation_model')!
  virtual void AssertType(Hphandle handle) const;

public:

  // Deep copy of all data represented by this object instance
  HVariationModel Clone() const;



/*****************************************************************************
 * Operator-based class constructors
 *****************************************************************************/

  // read_variation_model: Read a variation model from a file.
  explicit HVariationModel(const HString& FileName);

  // read_variation_model: Read a variation model from a file.
  explicit HVariationModel(const char* FileName);

#ifdef _WIN32
  // read_variation_model: Read a variation model from a file.
  explicit HVariationModel(const wchar_t* FileName);
#endif

  // create_variation_model: Create a variation model for image comparison.
  explicit HVariationModel(Hlong Width, Hlong Height, const HString& Type, const HString& Mode);

  // create_variation_model: Create a variation model for image comparison.
  explicit HVariationModel(Hlong Width, Hlong Height, const char* Type, const char* Mode);

#ifdef _WIN32
  // create_variation_model: Create a variation model for image comparison.
  explicit HVariationModel(Hlong Width, Hlong Height, const wchar_t* Type, const wchar_t* Mode);
#endif




  /***************************************************************************
   * Operators                                                               *
   ***************************************************************************/

  // Deserialize a variation model.
  void DeserializeVariationModel(const HSerializedItem& SerializedItemHandle);

  // Serialize a variation model.
  HSerializedItem SerializeVariationModel() const;

  // Read a variation model from a file.
  void ReadVariationModel(const HString& FileName);

  // Read a variation model from a file.
  void ReadVariationModel(const char* FileName);

#ifdef _WIN32
  // Read a variation model from a file.
  void ReadVariationModel(const wchar_t* FileName);
#endif

  // Write a variation model to a file.
  void WriteVariationModel(const HString& FileName) const;

  // Write a variation model to a file.
  void WriteVariationModel(const char* FileName) const;

#ifdef _WIN32
  // Write a variation model to a file.
  void WriteVariationModel(const wchar_t* FileName) const;
#endif

  // Return the threshold images used for image comparison by a variation model.
  HImage GetThreshImagesVariationModel(HImage* MaxImage) const;

  // Return the images used for image comparison by a variation model.
  HImage GetVariationModel(HImage* VarImage) const;

  // Compare an image to a variation model.
  HRegion CompareExtVariationModel(const HImage& Image, const HString& Mode) const;

  // Compare an image to a variation model.
  HRegion CompareExtVariationModel(const HImage& Image, const char* Mode) const;

#ifdef _WIN32
  // Compare an image to a variation model.
  HRegion CompareExtVariationModel(const HImage& Image, const wchar_t* Mode) const;
#endif

  // Compare an image to a variation model.
  HRegion CompareVariationModel(const HImage& Image) const;

  // Prepare a variation model for comparison with an image.
  void PrepareDirectVariationModel(const HImage& RefImage, const HImage& VarImage, const HTuple& AbsThreshold, const HTuple& VarThreshold) const;

  // Prepare a variation model for comparison with an image.
  void PrepareDirectVariationModel(const HImage& RefImage, const HImage& VarImage, double AbsThreshold, double VarThreshold) const;

  // Prepare a variation model for comparison with an image.
  void PrepareVariationModel(const HTuple& AbsThreshold, const HTuple& VarThreshold) const;

  // Prepare a variation model for comparison with an image.
  void PrepareVariationModel(double AbsThreshold, double VarThreshold) const;

  // Train a variation model.
  void TrainVariationModel(const HImage& Images) const;

  // Free the memory of a variation model.
  void ClearVariationModel() const;

  // Free the memory of the training data of a variation model.
  void ClearTrainDataVariationModel() const;

  // Create a variation model for image comparison.
  void CreateVariationModel(Hlong Width, Hlong Height, const HString& Type, const HString& Mode);

  // Create a variation model for image comparison.
  void CreateVariationModel(Hlong Width, Hlong Height, const char* Type, const char* Mode);

#ifdef _WIN32
  // Create a variation model for image comparison.
  void CreateVariationModel(Hlong Width, Hlong Height, const wchar_t* Type, const wchar_t* Mode);
#endif

};

// forward declarations and types for internal array implementation

template<class T> class HSmartPtr;
template<class T> class HHandleBaseArrayRef;

typedef HHandleBaseArrayRef<HVariationModel> HVariationModelArrayRef;
typedef HSmartPtr< HVariationModelArrayRef > HVariationModelArrayPtr;


// Represents multiple tool instances
class LIntExport HVariationModelArray : public HHandleBaseArray
{

public:

  // Create empty array
  HVariationModelArray();

  // Create array from native array of tool instances
  HVariationModelArray(HVariationModel* classes, Hlong length);

  // Copy constructor
  HVariationModelArray(const HVariationModelArray &tool_array);

  // Destructor
  virtual ~HVariationModelArray();

  // Assignment operator
  HVariationModelArray &operator=(const HVariationModelArray &tool_array);

  // Clears array and all tool instances
  virtual void Clear();

  // Get array of native tool instances
  const HVariationModel* Tools() const;

  // Get number of tools
  virtual Hlong Length() const;

  // Create tool array from tuple of handles
  virtual void SetFromTuple(const HTuple& handles);

  // Get tuple of handles for tool array
  virtual HTuple ConvertToTuple() const;

protected:

// Smart pointer to internal data container
   HVariationModelArrayPtr *mArrayPtr;
};

}

#endif
