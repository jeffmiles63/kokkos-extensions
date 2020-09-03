# kokkos-extensions
Plugins For Kokkos memory and execution spaces

## Instructions to use:

When configuring Kokkos using cmake populate the Kokkos_PLUGIN_PATH list with the paths to each plugin desired. 

For example: to enable the UmpireSpace plugin, use the following. 

`-DKokkos_PLUGIN_PATH="<local path to kokkos-extension>/UmpireSpace"`

To enable multiple plugins, separate paths with a semi-colon.

`-DKokkos_PLUGIN_PATH="<local path to kokkos-extensions>/UmpireSpace;<local path to kokkos-extensions>/SICMSpace"`

This option can also be used in conjunction with the kokkos generate_makefile.bash using the --cmake-flags option.  

`--cmake-flags="-DKokkos_PLUGIN_PATH=<local path to kokkos-extension>/UmpireSpace"`

-- Note that the Kokkos plugin mechanism only works with CMake builds

## Instruction for creating new plugins

The following files are required for a Kokkos plugin, but the contents may vary.

 - ` <plugin root>/CMakeLists.txt ` - CMake file used to configure CMake / Kokkos options necessary for plugin.
 - ` <plugin root>/cmake/LinkCoreTPL.cmake ` - CMake file used to add TPLs and packages.  The format may vary based on the tpls required, but ideally, linking TPLS with the KOKKOS_LINK_TPL command is best.
 - ` <plugin root>/core/src/CMakeLists.txt ` - CMake file used to include source files into the kokkoscore library.  This is also where the installation command for headers would be included.  Adding source files to the kokkoscore library is done by appending the source globs to the KOKKOS_CORE_SRCS list
 - ` <plugin root>/core/unit_test/CMakeLists.txt ` - CMake file used to include source files into kokkos unit tests.  This is usually accomplished by adding the test cpp files to the `<BackendName>_SOURCES` list.  If the plugin contains a new execution space, then the incremental tests should be utilized.
 - ` <plugin root>/core/src/fwd/Kokkos_Fwd_<space name>.hpp ` - header file containing forward declare of plugin space/s.  This is included in Kokkos_Core_fwd.hpp
 - ` <plugin root>/core/src/decl/Kokkos_Declare_<space name>.hpp ` - header file containing full declaration of plugin space/s.  This is included in Kokkos_Core.hpp.  Usually this is a redirect where it includes the actual space header file/s.
 - `<plugin root>/core/src/Kokkos_Post_Include_<space name>.hpp` - optional header file containing declarations that follow the general set of spaces, views, etc.
 - `<plugin root>/core/src/setup/Kokkos_Setup_<space name>.hpp` - optional header file containing backend pre-requisits (included before execution space declarations)
  
If the Plugin contains a Memory Space only, the name of the space must be appended to Kokkos option `KOKKOS_MEMSPACE_LIST`.   The name should match the `<space name>` referenced above.  
  
  `LIST(APPEND KOKKOS_MEMSPACE_LIST <space name>)`

If the Plugin contains an Execution Space, the name of the space must be appended to the Kokkos option `KOKKOS_ENABLED_DEVICES`.

`LIST(APPEND KOKKOS_ENABLED_DEVICES <space name>)`

If there are additional setup requirements for the execution backend, then the name must be appended to the `DEVICE_SETUP_LIST` option.  With this addition, the file ` <plugin root>/core/src/setup/Kokkos_Setup_<space name>.hpp ` is also required.

If there are additional definitions needed after the spaces are declared, the backend name must be appeded to the `KOKKOS_BACKEND_POST_INCLUDE_LIST` option.  With this addition the file `<plugin root>/core/src/Kokkos_Post_Include_<space name>.hpp` is also required.

## Initialization and Finalization

An implementation of the following virtual class is required to facilitate proxy calls from `Kokkos::initialize`, `Kokkos::finalize`, `Kokkos::fence` and `Kokkos::print_configuration`.

```c++
namespace Kokkos {
namespace Impl {
class MySpaceFactory : public ExecSpaceFactoryBase {
  public:
    MySpaceFactory();
    virtual ~MySpaceFactory();
    virtual void initialize(const InitArguments& args);
    virtual void finalize(const bool);
    virtual void fence();
    virtual void print_configuration(std::ostringstream& msg, const bool detail);
};
} // namespace Impl
} // namespace Kokkos
```

`initialize` is used to initialize the execution space resources.
`finalize` is used to finalize / close the execution space resources.
`fence` is the global (static) fence of the execution space resources.
`print_configuration` prints the runtime configuration for the execution space.

Within the constructor, the execution space should be registered with the ExecSpaceManager using the following.

`Kokkos::Impl::ExecSpaceManager::get_instance().register_space_factory("<three digit number>_<BackendName>", this);`

The name is comprised of a number and a name.  This is so the initialize functions can be called in a "known" order.  Serial uses `100_Serial`, host parallel backends are `< 100` and device (accelerator) backes are `> 100`.

in the destructor the execution space should be unregistered using the same name.

`Kokkos::Impl::ExecSpaceManager::get_instance().unregister_space_factory("<three digit number>_<BackendName>");`

## Space Defaults

The default execution and memoryspaces logic can only be set with internally loaded spaces; however the defaults can be overridden with several Kokkos options.  

 - Kokkos_DEFAULT_DEVICE_EXEC_SPACE - this will override the default device execution space.
 - Kokkos_DEFAULT_HOST_PARALLEL_EXEC_SPACE - this will override the default host parallel execution space.
 - Kokkos_DEFAULT_DEVICE_MEMORY_SPACE - this will override the default device memory space.
 - Kokkos_DEFAULT_HOST_MEMORY_SPACE - this will override the default host memory space.
