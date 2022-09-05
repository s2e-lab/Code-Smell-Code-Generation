def classify_comment(comment, cls=None):
    """
    If 'reported' class is provided no training occures, the comment's class
    is simply set as such and removed.

    If no class is provided a lookup is done to see if the comment has been
    reported by users as abusive. If indicated as abusive class is set
    as 'reported'and the comment being removed.
    """
    if cls not in ['spam', 'ham', 'unsure', 'reported', None]:
        raise Exception("Unrecognized classifications.")

    classified_comment, cr = models.ClassifiedComment.objects.get_or_create(
        comment=comment
    )

    if cls == 'spam' and classified_comment.cls != 'spam':
        comment.is_removed = True
        comment.save()
        classified_comment.cls = cls
        classified_comment.save()
        return classified_comment

    if cls == 'ham' and classified_comment.cls != 'ham':
        comment.is_removed = False
        comment.save()
        classified_comment.cls = cls
        classified_comment.save()
        return classified_comment

    if cls == 'unsure' and classified_comment.cls != 'unsure':
        classified_comment.cls = cls
        classified_comment.save()
        return classified_comment

    if cls == 'reported' and classified_comment.cls != 'reported':
        comment.is_removed = True
        comment.save()
        classified_comment.cls = cls
        classified_comment.save()
        return classified_comment

    if cls is None:
        cls = 'unsure'
        comment_content_type = ContentType.objects.get_for_model(comment)
        moderator_settings = getattr(settings, 'MODERATOR', DEFAULT_CONFIG)
        if Vote.objects.filter(
            content_type=comment_content_type,
            object_id=comment.id,
            vote=-1
        ).count() >= moderator_settings['ABUSE_CUTOFF']:
            cls = 'reported'
            comment.is_removed = True
            comment.save()
            classified_comment.cls = cls
            classified_comment.save()
            return classified_comment
        else:
            comment.is_removed = cls == 'spam'
            comment.save()
            classified_comment.cls = cls
            classified_comment.save()
            return classified_comment

    return classified_comment