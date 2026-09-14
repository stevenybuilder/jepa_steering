import unittest

from offline_study.tasks.droid.droid_fit_availability import camera_object, equivalent_metadata
from offline_study.tasks.droid.droid_fit_eligible import native_clip_possible


class AvailabilityTests(unittest.TestCase):
    def test_native_window_must_have_nonempty_end_frame_interval(self):
        self.assertFalse(native_clip_possible(60,60.))
        self.assertTrue(native_clip_possible(61,60.))
        self.assertFalse(native_clip_possible(23,60.))
        self.assertFalse(native_clip_possible(7,60.))
        with self.assertRaises(ValueError):native_clip_possible(80,float('nan'))

    def test_metadata_aliases_must_preserve_every_input_field(self):
        first={'uuid':'a','user_id':'x','left_mp4_path':'v.mp4','length':500}
        second={**first,'uuid':'b','user_id':'y'}
        self.assertEqual(equivalent_metadata([first,second]),first)
        self.assertEqual(equivalent_metadata([{**first,'lab':'TRI'},{**second,'lab':'tri'}]),{**first,'lab':'TRI'})
        with self.assertRaises(ValueError):equivalent_metadata([{**first,'lab':'TRI'},{**second,'lab':'other'}])
        for invalid in ([],[first]*5,[first,{**second,'left_mp4_path':'other.mp4'}],
                [first,{**second,'length':501}]):
            with self.assertRaises(ValueError):equivalent_metadata(invalid)

    def test_only_missing_exact_native_camera_is_excludable(self):
        meta={'left_mp4_path':'source/recordings/MP4/123.mp4'}
        obj={'name':'data/recordings/MP4/123.mp4','size':'100','md5Hash':'AAAAAAAAAAAAAAAAAAAAAA=='}
        self.assertEqual(camera_object('data/',{obj['name']:obj},meta),(obj,obj['name']))
        self.assertEqual(camera_object('data/',{},meta),(None,obj['name']))
        with self.assertRaises(ValueError):camera_object('data/',{obj['name']:{**obj,'size':'0'}},meta)
        with self.assertRaises(ValueError):camera_object('data/',{}, {'left_mp4_path':'recordings/MP4/../123.mp4'})
        with self.assertRaises(KeyError):camera_object('data/',{}, {})


if __name__=='__main__':unittest.main()
