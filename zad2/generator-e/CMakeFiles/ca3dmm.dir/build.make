# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wladekp/HPC/zad2/generator-e

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wladekp/HPC/zad2/generator-e

# Include any dependencies generated for this target.
include CMakeFiles/ca3dmm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ca3dmm.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ca3dmm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ca3dmm.dir/flags.make

CMakeFiles/ca3dmm.dir/ca3dmm.cpp.o: CMakeFiles/ca3dmm.dir/flags.make
CMakeFiles/ca3dmm.dir/ca3dmm.cpp.o: ca3dmm.cpp
CMakeFiles/ca3dmm.dir/ca3dmm.cpp.o: CMakeFiles/ca3dmm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wladekp/HPC/zad2/generator-e/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ca3dmm.dir/ca3dmm.cpp.o"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ca3dmm.dir/ca3dmm.cpp.o -MF CMakeFiles/ca3dmm.dir/ca3dmm.cpp.o.d -o CMakeFiles/ca3dmm.dir/ca3dmm.cpp.o -c /home/wladekp/HPC/zad2/generator-e/ca3dmm.cpp

CMakeFiles/ca3dmm.dir/ca3dmm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ca3dmm.dir/ca3dmm.cpp.i"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wladekp/HPC/zad2/generator-e/ca3dmm.cpp > CMakeFiles/ca3dmm.dir/ca3dmm.cpp.i

CMakeFiles/ca3dmm.dir/ca3dmm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ca3dmm.dir/ca3dmm.cpp.s"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wladekp/HPC/zad2/generator-e/ca3dmm.cpp -o CMakeFiles/ca3dmm.dir/ca3dmm.cpp.s

CMakeFiles/ca3dmm.dir/densematgen.cpp.o: CMakeFiles/ca3dmm.dir/flags.make
CMakeFiles/ca3dmm.dir/densematgen.cpp.o: densematgen.cpp
CMakeFiles/ca3dmm.dir/densematgen.cpp.o: CMakeFiles/ca3dmm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wladekp/HPC/zad2/generator-e/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ca3dmm.dir/densematgen.cpp.o"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ca3dmm.dir/densematgen.cpp.o -MF CMakeFiles/ca3dmm.dir/densematgen.cpp.o.d -o CMakeFiles/ca3dmm.dir/densematgen.cpp.o -c /home/wladekp/HPC/zad2/generator-e/densematgen.cpp

CMakeFiles/ca3dmm.dir/densematgen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ca3dmm.dir/densematgen.cpp.i"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wladekp/HPC/zad2/generator-e/densematgen.cpp > CMakeFiles/ca3dmm.dir/densematgen.cpp.i

CMakeFiles/ca3dmm.dir/densematgen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ca3dmm.dir/densematgen.cpp.s"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wladekp/HPC/zad2/generator-e/densematgen.cpp -o CMakeFiles/ca3dmm.dir/densematgen.cpp.s

CMakeFiles/ca3dmm.dir/multiplication.cpp.o: CMakeFiles/ca3dmm.dir/flags.make
CMakeFiles/ca3dmm.dir/multiplication.cpp.o: multiplication.cpp
CMakeFiles/ca3dmm.dir/multiplication.cpp.o: CMakeFiles/ca3dmm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wladekp/HPC/zad2/generator-e/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ca3dmm.dir/multiplication.cpp.o"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ca3dmm.dir/multiplication.cpp.o -MF CMakeFiles/ca3dmm.dir/multiplication.cpp.o.d -o CMakeFiles/ca3dmm.dir/multiplication.cpp.o -c /home/wladekp/HPC/zad2/generator-e/multiplication.cpp

CMakeFiles/ca3dmm.dir/multiplication.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ca3dmm.dir/multiplication.cpp.i"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wladekp/HPC/zad2/generator-e/multiplication.cpp > CMakeFiles/ca3dmm.dir/multiplication.cpp.i

CMakeFiles/ca3dmm.dir/multiplication.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ca3dmm.dir/multiplication.cpp.s"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wladekp/HPC/zad2/generator-e/multiplication.cpp -o CMakeFiles/ca3dmm.dir/multiplication.cpp.s

# Object files for target ca3dmm
ca3dmm_OBJECTS = \
"CMakeFiles/ca3dmm.dir/ca3dmm.cpp.o" \
"CMakeFiles/ca3dmm.dir/densematgen.cpp.o" \
"CMakeFiles/ca3dmm.dir/multiplication.cpp.o"

# External object files for target ca3dmm
ca3dmm_EXTERNAL_OBJECTS =

ca3dmm: CMakeFiles/ca3dmm.dir/ca3dmm.cpp.o
ca3dmm: CMakeFiles/ca3dmm.dir/densematgen.cpp.o
ca3dmm: CMakeFiles/ca3dmm.dir/multiplication.cpp.o
ca3dmm: CMakeFiles/ca3dmm.dir/build.make
ca3dmm: CMakeFiles/ca3dmm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wladekp/HPC/zad2/generator-e/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ca3dmm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ca3dmm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ca3dmm.dir/build: ca3dmm
.PHONY : CMakeFiles/ca3dmm.dir/build

CMakeFiles/ca3dmm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ca3dmm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ca3dmm.dir/clean

CMakeFiles/ca3dmm.dir/depend:
	cd /home/wladekp/HPC/zad2/generator-e && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wladekp/HPC/zad2/generator-e /home/wladekp/HPC/zad2/generator-e /home/wladekp/HPC/zad2/generator-e /home/wladekp/HPC/zad2/generator-e /home/wladekp/HPC/zad2/generator-e/CMakeFiles/ca3dmm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ca3dmm.dir/depend

