def save(self):
        """ Generate a random username before falling back to parent signup form """
        while True:
            username = sha1(str(random.random()).encode('utf-8')).hexdigest()[:5]
            try:
                get_user_model().objects.get(username__iexact=username)
            except get_user_model().DoesNotExist: break

        self.cleaned_data['username'] = username
        return super(SignupFormOnlyEmail, self).save()