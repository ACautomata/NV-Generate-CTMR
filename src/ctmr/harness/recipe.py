# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thin forwarding shim -- the pinned-recipe guards moved to ``ctmr.domain.recipe``
(ADR-0015 §2 domain layering, #133). The scripts-side consumers keep importing
from here until the application batch relocates them; behaviour is unchanged.
"""

from ctmr.domain.recipe import CrossModalRecipeSpec, P1RecipeSpec, P2RecipeSpec

__all__ = ["P1RecipeSpec", "P2RecipeSpec", "CrossModalRecipeSpec"]
