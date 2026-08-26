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

"""Thin re-export of the pinned-recipe guards -- they moved to ``ctmr.domain.recipe``
(#133, ADR-0015 M2). The stage entries import from here; new code imports
``ctmr.domain.recipe`` directly."""

from ctmr.domain.recipe import P1RecipeSpec, P2RecipeSpec, P3RecipeSpec

__all__ = ["P1RecipeSpec", "P2RecipeSpec", "P3RecipeSpec"]
