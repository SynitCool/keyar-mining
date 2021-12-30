from functools import wraps
import time
import dill


class Wrapper:
    """
    Wrapper
    """

    def __init__(self):
        self.__step = []
        self.__func_map = {}

    def save_func(self, name):
        """
        Saving function to wrapper

        Parameters
        ----------
        name: str
            The name of the function.

        """

        # decorator
        def get_func(func):
            @wraps(func)
            def saving_func():
                print(f"{func.__name__} is saved with name {name}")

            print(f"{func.__name__} is saved")
            self.__func_map[name] = func

            return saving_func

        return get_func

    def save_object_to_file(self, object_path):
        """
        Saving the wrapper to file

        Parameters
        ----------
        object_path: str or path
            The path to save the wrapper.

        """

        # write binary the object
        with open(object_path, "wb") as file:
            dill.dump(self, file)

    def save_object_to_var(self):
        """
        Saving the wrapper to variable

        Returns
        -------
        object: binary
            The object type dump to binary.

        """

        return dill.dumps(self)

    def load_func(self, *args):
        """
        Load the the wrapper based the step

        Parameters
        ----------
        *args: any type
            The parameters on first step function.

        """

        # check if has no step
        if len(self.__step) == 0:
            raise ValueError("Specify the step parameter")

        # signal before execution
        print("----------------\nBefore execution\n----------------")

        # iterating through the step
        for i in range(len(self.__step)):
            # check in the first step
            if i == 0:
                # get the time to execute the function
                start_time = time.time()
                output = self.__func_map[self.__step[i]](*args)
                print(
                    f"- Time to execute {self.__func_map[self.__step[i]].__name__} function: {time.time() - start_time}"
                )
                continue
            # check in the last step
            elif i == len(self.__step) - 1:
                # check if the error has different parameter
                try:
                    # get the time to execute the function
                    if type(output) == tuple:
                        start_time = time.time()
                        output = self.__func_map[self.__step[i]](*output)
                    else:
                        start_time = time.time()
                        output = self.__func_map[self.__step[i]](output)

                    print(
                        f"- Time to execute {self.__func_map[self.__step[i]].__name__} function: {time.time() - start_time}"
                    )

                    # check after execution
                    print("---------------\nAfter execution\n---------------")
                    return output
                except TypeError:
                    # get the time to execute the function
                    start_time = time.time()
                    output = self.__func_map[self.__step[i]]()
                    print(
                        f"- Time to execute {self.__func_map[self.__step[i]].__name__} function: {time.time() - start_time}"
                    )

                    # check after execution
                    print("---------------\nAfter execution\n---------------")
                    return output

            # execute the function based on step
            try:
                if type(output) == tuple:
                    start_time = time.time()
                    output = self.__func_map[self.__step[i]](*output)

                    print(
                        f"- Time to execute {self.__func_map[self.__step[i]].__name__} function: {time.time() - start_time}"
                    )
                else:
                    start_time = time.time()
                    output = self.__func_map[self.__step[i]](output)

                    print(
                        f"- Time to execute {self.__func_map[self.__step[i]].__name__} function: {time.time() - start_time}"
                    )
            except TypeError:
                output = self.__func_map[self.__step[i]]()

                print(
                    f"- Time to execute {self.__func_map[self.__step[i]].__name__} function: {time.time() - start_time}"
                )

        # check after execution
        print("---------------\nAfter execution\n---------------")

    def add_step(self, new_step):
        """
        Adding step

        Parameters
        ----------
        new_step: str
            The step for the object to load.
        """

        # check if new_step in func_map
        if new_step in self.__func_map:
            self.__step.append(new_step)
        else:
            raise ValueError("Add step based on the function that saved")

    @property
    def get_data_func(self):
        """
        Getter for function
        """

        # print the func_map if has func_map
        print("Data Function Structure\n=======================")
        if self.__func_map:
            for key in self.__func_map:
                print(f"- name AS {key}, func AS {self.__func_map[key].__name__}")
        else:
            print("- There are no data yet -")

    @property
    def get_step(self):
        """
        Getter for step
        """

        # print the step if has step
        print("Step Structure\n==============")
        if self.__step:
            for i, name_step in enumerate(self.__step, 1):
                print(f"{i}. {name_step}")
        else:
            print("- There are no step yet -")
