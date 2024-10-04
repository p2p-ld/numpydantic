# testing

Utilities for testing and 3rd-party interface development.

Only things that *don't* require pytest go in this module. 
We want to keep all test-time specific behavior there,
and have this just serve as helpers exposed for downstream interface development.

We want to avoid pytest stuff bleeding in here because then we limit
the ability for downstream developers to configure their own tests.

*(If there is some reason to change this division of labor, just raise an issue and let's chat.)*

```{toctree}
cases
helpers
```