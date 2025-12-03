from typing import Callable, Any


class Container:
    """ A simple dependency injection container
    to manage and resolve dependencies of pipelines and components. 
    """

    def __init__(self):
        """ Initializes the dependency injection container. """
        self.__providers: dict[str, Callable[[], Any]] = {}
        self.__singletons: dict[str, Any] = {}


    def register(self, name: str, provider: Callable[[], Any], singleton: bool = False):
        """ Registers a provider function for a dependency.

        Parameters
        ----------
        name : str
            The name of the dependency.
        provider : Callable[[], Any]
            A function that provides an instance of the dependency.
        singleton : bool, optional
            If True, the same instance will be returned on each request, by default False.
        """
        self.__providers[name] = (provider, singleton)


    def resolve(self, name: str):
        """ Resolves a dependency by its name.

        Parameters
        ----------
        name : str
            The name of the dependency to resolve.

        Returns
        -------
        Any
            An instance of the requested dependency.

        Raises
        ------
        ValueError
            If the dependency is not registered.
        """
        if name not in self.__providers:
            raise ValueError(f"Dependency '{name}' is not registered.")

        provider, singleton = self.__providers[name]

        if singleton:
            if name not in self.__singletons:
                self.__singletons[name] = provider()
            return self.__singletons[name]
        else:
            return provider()