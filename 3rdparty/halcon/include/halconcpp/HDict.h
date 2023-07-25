/***********************************************************
 * File generated by the HALCON-Compiler hcomp version 20.11
 * Usage: Interface to C++
 *
 * Software by: MVTec Software GmbH, www.mvtec.com
 ***********************************************************/


#ifndef HCPP_HDICT
#define HCPP_HDICT

namespace HalconCpp
{

// Represents an instance of a Dictionary.
class LIntExport HDict : public HHandle
{

public:

  // Copy constructor
  HDict(const HDict& source) : HHandle(source) {}

  // Copy constructor
  HDict(const HHandle& handle);

  // Create HDict from handle, taking ownership
  explicit HDict(Hlong handle);

  bool operator==(const HHandle& obj) const
  {
    return HHandleBase::operator==(obj);
  }

  bool operator!=(const HHandle& obj) const
  {
    return HHandleBase::operator!=(obj);
  }

protected:

  // Verify matching semantic type ('dict')!
  virtual void AssertType(Hphandle handle) const;

public:



/*****************************************************************************
 * Operator-based class constructors
 *****************************************************************************/

  // create_dict: Create a new empty dictionary.
  explicit HDict();

  // read_dict: Read a dictionary from a file.
  explicit HDict(const HString& FileName, const HTuple& GenParamName, const HTuple& GenParamValue);

  // read_dict: Read a dictionary from a file.
  explicit HDict(const HString& FileName, const HString& GenParamName, const HString& GenParamValue);

  // read_dict: Read a dictionary from a file.
  explicit HDict(const char* FileName, const char* GenParamName, const char* GenParamValue);

#ifdef _WIN32
  // read_dict: Read a dictionary from a file.
  explicit HDict(const wchar_t* FileName, const wchar_t* GenParamName, const wchar_t* GenParamValue);
#endif




  /***************************************************************************
   * Operators                                                               *
   ***************************************************************************/

  // Copy a dictionary.
  HDict CopyDict(const HTuple& GenParamName, const HTuple& GenParamValue) const;

  // Copy a dictionary.
  HDict CopyDict(const HString& GenParamName, const HString& GenParamValue) const;

  // Copy a dictionary.
  HDict CopyDict(const char* GenParamName, const char* GenParamValue) const;

#ifdef _WIN32
  // Copy a dictionary.
  HDict CopyDict(const wchar_t* GenParamName, const wchar_t* GenParamValue) const;
#endif

  // Create a new empty dictionary.
  void CreateDict();

  // Retrieve an object associated with the key from the dictionary.
  HObject GetDictObject(const HTuple& Key) const;

  // Retrieve an object associated with the key from the dictionary.
  HObject GetDictObject(const HString& Key) const;

  // Retrieve an object associated with the key from the dictionary.
  HObject GetDictObject(const char* Key) const;

#ifdef _WIN32
  // Retrieve an object associated with the key from the dictionary.
  HObject GetDictObject(const wchar_t* Key) const;
#endif

  // Query dictionary parameters or information about a dictionary.
  HTuple GetDictParam(const HString& GenParamName, const HTuple& Key) const;

  // Query dictionary parameters or information about a dictionary.
  HTuple GetDictParam(const HString& GenParamName, const HString& Key) const;

  // Query dictionary parameters or information about a dictionary.
  HTuple GetDictParam(const char* GenParamName, const char* Key) const;

#ifdef _WIN32
  // Query dictionary parameters or information about a dictionary.
  HTuple GetDictParam(const wchar_t* GenParamName, const wchar_t* Key) const;
#endif

  // Retrieve a tuple associated with the key from the dictionary.
  HTuple GetDictTuple(const HTuple& Key) const;

  // Retrieve a tuple associated with the key from the dictionary.
  HTuple GetDictTuple(const HString& Key) const;

  // Retrieve a tuple associated with the key from the dictionary.
  HTuple GetDictTuple(const char* Key) const;

#ifdef _WIN32
  // Retrieve a tuple associated with the key from the dictionary.
  HTuple GetDictTuple(const wchar_t* Key) const;
#endif

  // Read a dictionary from a file.
  void ReadDict(const HString& FileName, const HTuple& GenParamName, const HTuple& GenParamValue);

  // Read a dictionary from a file.
  void ReadDict(const HString& FileName, const HString& GenParamName, const HString& GenParamValue);

  // Read a dictionary from a file.
  void ReadDict(const char* FileName, const char* GenParamName, const char* GenParamValue);

#ifdef _WIN32
  // Read a dictionary from a file.
  void ReadDict(const wchar_t* FileName, const wchar_t* GenParamName, const wchar_t* GenParamValue);
#endif

  // Remove keys from a dictionary.
  void RemoveDictKey(const HTuple& Key) const;

  // Remove keys from a dictionary.
  void RemoveDictKey(const HString& Key) const;

  // Remove keys from a dictionary.
  void RemoveDictKey(const char* Key) const;

#ifdef _WIN32
  // Remove keys from a dictionary.
  void RemoveDictKey(const wchar_t* Key) const;
#endif

  // Add a key/object pair to the dictionary.
  void SetDictObject(const HObject& Object, const HTuple& Key) const;

  // Add a key/object pair to the dictionary.
  void SetDictObject(const HObject& Object, const HString& Key) const;

  // Add a key/object pair to the dictionary.
  void SetDictObject(const HObject& Object, const char* Key) const;

#ifdef _WIN32
  // Add a key/object pair to the dictionary.
  void SetDictObject(const HObject& Object, const wchar_t* Key) const;
#endif

  // Add a key/tuple pair to the dictionary.
  void SetDictTuple(const HTuple& Key, const HTuple& Tuple) const;

  // Add a key/tuple pair to the dictionary.
  void SetDictTuple(const HString& Key, const HTuple& Tuple) const;

  // Add a key/tuple pair to the dictionary.
  void SetDictTuple(const char* Key, const HTuple& Tuple) const;

#ifdef _WIN32
  // Add a key/tuple pair to the dictionary.
  void SetDictTuple(const wchar_t* Key, const HTuple& Tuple) const;
#endif

  // Write a dictionary to a file.
  void WriteDict(const HString& FileName, const HTuple& GenParamName, const HTuple& GenParamValue) const;

  // Write a dictionary to a file.
  void WriteDict(const HString& FileName, const HString& GenParamName, const HString& GenParamValue) const;

  // Write a dictionary to a file.
  void WriteDict(const char* FileName, const char* GenParamName, const char* GenParamValue) const;

#ifdef _WIN32
  // Write a dictionary to a file.
  void WriteDict(const wchar_t* FileName, const wchar_t* GenParamName, const wchar_t* GenParamValue) const;
#endif

};

// forward declarations and types for internal array implementation

template<class T> class HSmartPtr;
template<class T> class HHandleBaseArrayRef;

typedef HHandleBaseArrayRef<HDict> HDictArrayRef;
typedef HSmartPtr< HDictArrayRef > HDictArrayPtr;


// Represents multiple tool instances
class LIntExport HDictArray : public HHandleBaseArray
{

public:

  // Create empty array
  HDictArray();

  // Create array from native array of tool instances
  HDictArray(HDict* classes, Hlong length);

  // Copy constructor
  HDictArray(const HDictArray &tool_array);

  // Destructor
  virtual ~HDictArray();

  // Assignment operator
  HDictArray &operator=(const HDictArray &tool_array);

  // Clears array and all tool instances
  virtual void Clear();

  // Get array of native tool instances
  const HDict* Tools() const;

  // Get number of tools
  virtual Hlong Length() const;

  // Create tool array from tuple of handles
  virtual void SetFromTuple(const HTuple& handles);

  // Get tuple of handles for tool array
  virtual HTuple ConvertToTuple() const;

protected:

// Smart pointer to internal data container
   HDictArrayPtr *mArrayPtr;
};

}

#endif
