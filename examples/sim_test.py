import mujoco
import mediapy as media


xml = """
<mujoco model="2-link 6-muscle arm">
    <!--  Copyright © 2018, Roboti LLC

          This file is licensed under the MuJoCo Resource License (the "License").
          You may not use this file except in compliance with the License.
          You may obtain a copy of the License at

            https://www.roboti.us/resourcelicense.txt
    -->

    <option timestep="0.005" iterations="50" solver="Newton" tolerance="1e-10"/>

    <size njmax="50" nconmax="10" nstack="200"/>

    <visual>
        <rgba haze=".3 .3 .3 1"/>
    </visual>

    <default>
        <joint type="hinge" pos="0 0 0" axis="0 0 1" limited="true" range="0 120" damping="0.1"/>
        <muscle ctrllimited="true" ctrlrange="0 1"/>
    </default>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.6 0.6 0.6" rgb2="0 0 0" width="512" height="512"/> 

        <texture name="texplane" type="2d" builtin="checker" rgb1=".25 .25 .25" rgb2=".3 .3 .3" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>  

        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
    </asset>

    <worldbody>
        <geom name="floor" pos="0 0 -0.5" size="0 0 1" type="plane" material="matplane"/>

        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>

        <site name="s0" pos="-0.15 0 0" size="0.02"/>
        <site name="x0" pos="0 -0.15 0" size="0.02" rgba="0 .7 0 1" group="1"/>

        <body pos="0 0 0">
            <geom name="upper arm" type="capsule" size="0.045" fromto="0 0 0  0.5 0 0" rgba=".5 .1 .1 1"/>
            <joint name="shoulder"/>
            <geom name="shoulder" type="cylinder" pos="0 0 0" size=".1 .05" rgba=".5 .1 .8 .5" mass="0" group="1"/>

            <site name="s1" pos="0.15 0.06 0" size="0.02"/>
            <site name="s2" pos="0.15 -0.06 0" size="0.02"/>
            <site name="s3" pos="0.4 0.06 0" size="0.02"/>
            <site name="s4" pos="0.4 -0.06 0" size="0.02"/>
            <site name="s5" pos="0.25 0.1 0" size="0.02"/>
            <site name="s6" pos="0.25 -0.1 0" size="0.02"/>
            <site name="x1" pos="0.5 -0.15 0" size="0.02" rgba="0 .7 0 1" group="1"/>

            <body pos="0.5 0 0">
                <geom name="forearm" type="capsule" size="0.035" fromto="0 0 0  0.5 0 0" rgba=".5 .1 .1 1"/>
                <joint name="elbow"/>
                <geom name="elbow" type="cylinder" pos="0 0 0" size=".08 .05" rgba=".5 .1 .8 .5" mass="0" group="1"/>

                <site name="s7" pos="0.11 0.05 0" size="0.02"/>
                <site name="s8" pos="0.11 -0.05 0" size="0.02"/>
            </body>
        </body>
    </worldbody>

    <tendon>
        <spatial name="SF" width="0.01">
            <site site="s0"/>
            <geom geom="shoulder"/>
            <site site="s1"/>
        </spatial>

        <spatial name="SE" width="0.01">
            <site site="s0"/>
            <geom geom="shoulder" sidesite="x0"/>
            <site site="s2"/>
        </spatial>

        <spatial name="EF" width="0.01">
            <site site="s3"/>
            <geom geom="elbow"/>
            <site site="s7"/>
        </spatial>

        <spatial name="EE" width="0.01">
            <site site="s4"/>
            <geom geom="elbow" sidesite="x1"/>
            <site site="s8"/>
        </spatial>

        <spatial name="BF" width="0.009" rgba=".4 .6 .4 1">
            <site site="s0"/>
            <geom geom="shoulder"/>
            <site site="s5"/>
            <geom geom="elbow"/>
            <site site="s7"/>
        </spatial>

        <spatial name="BE" width="0.009" rgba=".4 .6 .4 1">
            <site site="s0"/>
            <geom geom="shoulder" sidesite="x0"/>
            <site site="s6"/>
            <geom geom="elbow" sidesite="x1"/>
            <site site="s8"/>
        </spatial>
    </tendon>   

    <actuator>
        <muscle name="SF" tendon="SF"/>
        <muscle name="SE" tendon="SE"/>
        <muscle name="EF" tendon="EF"/>
        <muscle name="EE" tendon="EE"/>
        <muscle name="BF" tendon="BF"/>
        <muscle name="BE" tendon="BE"/>
    </actuator>
</mujoco>

"""
# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Make renderer, render and show the pixels
renderer = mujoco.Renderer(model)
media.show_image(renderer.render())
