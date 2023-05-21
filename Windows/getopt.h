#pragma once
# ifndef __GETOPT_H_
# define __GETOPT_H_

# ifdef _GETOPT_API
#     undef _GETOPT_API
# endif
//------------------------------------------------------------------------------
# if defined(EXPORTS_GETOPT) && defined(STATIC_GETOPT)
#     error "The preprocessor definitions of EXPORTS_GETOPT and STATIC_GETOPT \
can only be used individually"
# elif defined(STATIC_GETOPT)
#     pragma message("Warning static builds of getopt violate the Lesser GNU \
Public License")
#     define _GETOPT_API
# elif defined(EXPORTS_GETOPT)
#     pragma message("Exporting getopt library")
#     define _GETOPT_API __declspec(dllexport)
# else
#     pragma message("Importing getopt library")
#     define _GETOPT_API __declspec(dllimport)
# endif

# include <tchar.h>
// Standard GNU options
# define null_argument           0 /*Argument Null*/
# define no_argument             0 /*Argument Switch Only*/
# define required_argument       1 /*Argument Required*/
# define optional_argument       2 /*Argument Optional*/
// Shorter Versions of options
# define ARG_NULL 0 /*Argument Null*/
# define ARG_NONE 0 /*Argument Switch Only*/
# define ARG_REQ  1 /*Argument Required*/
# define ARG_OPT  2 /*Argument Optional*/
// Change behavior for C\C++
# ifdef __cplusplus
#     define _BEGIN_EXTERN_C extern "C" {
#     define _END_EXTERN_C }
#     define _GETOPT_THROW throw()
# else
#     define _BEGIN_EXTERN_C
#     define _END_EXTERN_C
#     define _GETOPT_THROW
# endif
_BEGIN_EXTERN_C
extern _GETOPT_API TCHAR* optarg;
extern _GETOPT_API int    optind;
extern _GETOPT_API int    opterr;
extern _GETOPT_API int    optopt;
struct option
{
    /* The predefined macro variable __STDC__ is defined for C++, and it has the in-
       teger value 0 when it is used in an #if statement, indicating that the C++ l-
       anguage is not a proper superset of C, and that the compiler does not confor-
       m to C. In C, __STDC__ has the integer value 1. */
# if defined (__STDC__) && __STDC__
    const TCHAR* name;
# else
    TCHAR* name;
# endif
    int has_arg;
    int* flag;
    TCHAR val;
};
extern _GETOPT_API int getopt(int argc, TCHAR* const* argv
    , const TCHAR* optstring) _GETOPT_THROW;
extern _GETOPT_API int getopt_long
(int ___argc, TCHAR* const* ___argv
    , const TCHAR* __shortopts
    , const struct option* __longopts
    , int* __longind) _GETOPT_THROW;
extern _GETOPT_API int getopt_long_only
(int ___argc, TCHAR* const* ___argv
    , const TCHAR* __shortopts
    , const struct option* __longopts
    , int* __longind) _GETOPT_THROW;
// harly.he add for reentrant 12.09/2013
extern _GETOPT_API void getopt_reset() _GETOPT_THROW;
_END_EXTERN_C
// Undefine so the macros are not included
# undef _BEGIN_EXTERN_C
# undef _END_EXTERN_C
# undef _GETOPT_THROW
# undef _GETOPT_API
# endif  // __GETOPT_H_
