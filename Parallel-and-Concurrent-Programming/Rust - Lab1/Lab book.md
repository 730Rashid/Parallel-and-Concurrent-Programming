# 500083 Lab Book

## Week 1 - Lab A

28 Jan 2026

### Q1. Hello World

**Question:**

Locate the Solution Explorer within Visual Studio and select the HelloWorld project.

Right click on this project and select Build. This should compile and link the project.

Now run the HelloWorld program.

Change between Debug and Release mode. Compile again and rerun the program.

**Solution:**

```c++
#include <iostream>

int main(int, char**) {
   std::cout << "Hello World" << std::endl;
   return 0;
}
```

**Test data:**
"Hello World"


**Sample output:**

*Hello World*

**Reflection:**

*Reflect on what you have learnt from this exercise.*
-I learnt how to a create folder called "hello_world" and opened up a new terminal to cd inside the folder. I inspected the .rust file where main function has aready been defined with the "Hello_world" output in it and i ran the program using "cargo init" in the terminal.  

*Did you make any mistakes?*
-I accidently entered "cargo init" in the terminal when trying to run my program instead of "cargo run". A minior mistake but I realised you cant initialise cargo within it.


*In what way has your knowledge improved?*
-I now understand how to print "Hello World" in Rust and how to exeucte the script using cargo init in the terminal. I also learnt that Rust is a language similar to C++ except it is very strict in terms of data. for example if you initialise a variable it is a const unless you write "mut" just before declaring the variable name. 

**Questions:**

*Is there anything you would like to ask?*
No

### Q2. Console Window

"Hello, World!"
